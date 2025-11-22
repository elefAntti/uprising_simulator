class ScoreEstimator:
    def __init__(self, size=1.5):
        self._size = size
    def _quantify(self, cores):
        A = 0
        Neutral = 0
        B = 0
        for core in cores:
            z = (core[0] + core[1])
            if z < 0.5:
                A += 1
            elif z > 1.0:
                B += 1
            else:
                Neutral += 1
        return (A, Neutral, B)
    def _scored(self, old, new):
        scored_A = min(min(old[0] - new[0], 0) - min(new[1] - old[1], 0), 0)
        scored_B = min(min(old[2] - new[2], 0) - min(new[1] - old[1], 0), 0)
        return scoredA, scoredB

    def observe(self, red_cores, green_cores):
        pass
