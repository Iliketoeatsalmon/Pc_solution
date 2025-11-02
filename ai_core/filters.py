# ai_core/filters.py
import math
import time

class Kalman1D:
    """
    คาลแมนฟิลเตอร์แบบ 1 มิติ (สถานะ = ระยะ, สมมติความเร็วคงที่โดยประมาณ)
    x = [d]  (ใช้แบบง่ายที่สุด; ถ้าต้องการแบบมีความเร็ว เติมเป็น x=[d, v] ก็ได้)
    """
    def __init__(self, x0=100.0, p0=100.0, q=2.0, r=50.0):
        # x: estimate distance, P: covariance
        self.x = float(x0)
        self.P = float(p0)
        self.Q = float(q)   # process noise
        self.R = float(r)   # measurement noise
        self._last_t = None

    def reset(self, x0=None, p0=None):
        if x0 is not None: self.x = float(x0)
        if p0 is not None: self.P = float(p0)
        self._last_t = None

    def update(self, z, dt=None):
        """ z = measured distance (cm), dt optional (ไม่ระบุก็จะเดาเอง ~ frame time) """
        t = time.time()
        if dt is None:
            if self._last_t is None:
                dt = 0.033
            else:
                dt = max(1e-3, t - self._last_t)

        # 1) Predict (โมเดลคงที่: d_k = d_{k-1}, P เพิ่มด้วย Q)
        x_pred = self.x
        P_pred = self.P + self.Q

        # 2) Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (float(z) - x_pred)
        self.P = (1.0 - K) * P_pred

        self._last_t = t
        return self.x
