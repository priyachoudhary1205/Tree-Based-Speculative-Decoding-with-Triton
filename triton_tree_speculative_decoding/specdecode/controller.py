class DraftController:
    def __init__(self, min_depth=2, max_depth=12, initial=4):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth = initial

    def update(self, acceptance_rate: float, verify_ms: float, target_ms: float = 8.0):
        # Favor deeper drafts when acceptance is high and verification has budget.
        if acceptance_rate > 0.75 and verify_ms <= target_ms:
            self.depth = min(self.max_depth, self.depth + 1)
        elif acceptance_rate < 0.45 or verify_ms > target_ms * 1.25:
            self.depth = max(self.min_depth, self.depth - 1)
        return self.depth
