from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_ids = {config.eos} if isinstance(config.eos, int) else set(config.eos or ())
        self.block_size = config.kvcache_block_size
        self.prefix_cache_disabled_reason = config.prefix_cache_disabled_reason
        uses_state = self.prefix_cache_disabled_reason is not None
        self.free_state_slots = deque(range(self.max_num_seqs)) if uses_state else None
        self.block_manager = (BlockManager(config.num_kvcache_blocks, self.block_size, reuse_prefix=not uses_state)
                              if config.num_kvcache_blocks else None)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            new_request = not seq.block_table and seq.state_slot is None
            if new_request and self.free_state_slots is not None and not self.free_state_slots:
                break
            if new_request and self.block_manager is not None:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if new_request:
                if self.block_manager is not None:
                    self.block_manager.allocate(seq, num_cached_blocks)
                if self.free_state_slots is not None:
                    seq.state_slot = self.free_state_slots.popleft()
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while self.block_manager is not None and not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                if self.block_manager is not None:
                    self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self._release(seq)
        self.waiting.appendleft(seq)

    def _release(self, seq: Sequence):
        if self.block_manager is not None:
            self.block_manager.deallocate(seq)
        if seq.state_slot is not None:
            self.free_state_slots.append(seq.state_slot)
            seq.state_slot = None
        seq.num_cached_tokens = 0

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if self.block_manager is not None:
                self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id in self.eos_token_ids) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self._release(seq)
                self.running.remove(seq)
