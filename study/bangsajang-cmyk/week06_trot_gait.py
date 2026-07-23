import math
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class TrottingGait:
    """SpotMicro의 간단한 제자리 Trot 보행 궤적을 생성한다."""

    def __init__(self) -> None:
        # 보행 각 구간 시간, 단위: 밀리초
        self.t0 = 300
        self.t1 = 1200
        self.t2 = 300
        self.t3 = 200

        # Swing 구간에서 발을 들어 올리는 높이
        self.step_height = 30.0

        # 네 발의 기본 위치
        self.default_positions = {
            "FL": [100.0, -150.0, 80.0],
            "FR": [100.0, -150.0, -80.0],
            "RL": [-100.0, -150.0, 80.0],
            "RR": [-100.0, -150.0, -80.0],
        }

    @property
    def total_period(self) -> float:
        """전체 보행 주기를 반환한다."""
        return self.t0 + self.t1 + self.t2 + self.t3

    def calculate_leg_position(
        self,
        elapsed_ms: float,
        initial_position: list[float],
    ) -> list[float]:
        """현재 시간에 따른 다리 하나의 발끝 위치를 계산한다."""

        x, y, z = initial_position

        # 시간이 계속 증가해도 한 보행 주기 안의 값으로 변환
        elapsed_ms %= self.total_period

        # 1단계: 시작 위치에서 잠시 대기
        if elapsed_ms < self.t0:
            return [x, y, z]

        # 2단계: Stance phase
        # 발이 지면에 닿은 상태로 앞쪽에서 뒤쪽으로 이동
        if elapsed_ms < self.t0 + self.t1:
            progress = (elapsed_ms - self.t0) / self.t1

            x_offset = 40.0 - 80.0 * progress

            return [x + x_offset, y, z]

        # 3단계: 뒤쪽 위치에서 잠시 대기
        if elapsed_ms < self.t0 + self.t1 + self.t2:
            return [x - 40.0, y, z]

        # 4단계: Swing phase
        # 발을 들어 올리면서 앞쪽으로 복귀
        swing_time = elapsed_ms - self.t0 - self.t1 - self.t2
        progress = swing_time / self.t3

        x_offset = -40.0 + 80.0 * progress

        # 사인 함수를 사용해 발이 부드럽게 올라갔다 내려오도록 함
        y_offset = self.step_height * math.sin(math.pi * progress)

        return [x + x_offset, y + y_offset, z]

    def positions(self, current_time: float) -> dict[str, list[float]]:
        """네 발의 현재 위치를 계산한다."""

        elapsed_ms = current_time * 1000.0
        half_period = self.total_period / 2.0

        # FL과 RR에 적용되는 시간
        phase_a = elapsed_ms % self.total_period

        # FR과 RL에는 전체 주기의 절반만큼 시간차 적용
        phase_b = (elapsed_ms + half_period) % self.total_period

        return {
            "FL": self.calculate_leg_position(
                phase_a,
                self.default_positions["FL"],
            ),
            "RR": self.calculate_leg_position(
                phase_a,
                self.default_positions["RR"],
            ),
            "FR": self.calculate_leg_position(
                phase_b,
                self.default_positions["FR"],
            ),
            "RL": self.calculate_leg_position(
                phase_b,
                self.default_positions["RL"],
            ),
        }


# Trot 보행 객체 생성
gait = TrottingGait()

# Matplotlib 그래프 생성
figure, axis = plt.subplots()

axis.set_xlim(-160, 160)
axis.set_ylim(-190, -90)

axis.set_xlabel("X position")
axis.set_ylabel("Y position")
axis.set_title("Week06 - In-place Trot Gait")

# 네 발을 각각 점으로 생성
points = {
    "FL": axis.plot([], [], "o", label="FL")[0],
    "FR": axis.plot([], [], "o", label="FR")[0],
    "RL": axis.plot([], [], "o", label="RL")[0],
    "RR": axis.plot([], [], "o", label="RR")[0],
}

axis.legend()


def update(_: int):
    """애니메이션 프레임마다 네 발의 위치를 갱신한다."""

    current_time = time.time()
    positions = gait.positions(current_time)

    for leg_name, point in points.items():
        x, y, _ = positions[leg_name]

        point.set_data([x], [y])

    return list(points.values())


# 20밀리초마다 update 함수를 실행
animation = FuncAnimation(
    figure,
    update,
    interval=20,
    blit=True,
    cache_frame_data=False,
)

# 그래프 창 표시
plt.show()