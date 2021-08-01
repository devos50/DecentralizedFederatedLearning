package nl.tudelft.dfl.types

import org.nd4j.linalg.schedule.ISchedule
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType

enum class LearningRate(val id: String, val text: String, val schedule: ISchedule) {
    RATE_1EM3(
        "rate_1em3",
        "1e-3", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 1e-3))
    ),
    RATE_5EM2(
        "rate_5em2",
        "5e-2", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 0.05))
    ),
    SCHEDULE1(
        "schedule1",
        "{0 -> 0.06|100 -> 0.05|200 -> 0.028|300 -> 0.006|400 -> 0.001", MapSchedule(
            ScheduleType.ITERATION, hashMapOf(
                0 to 0.06,
                100 to 0.05,
                200 to 0.028,
                300 to 0.006,
                400 to 0.001
            )
        )
    );

    companion object {
        fun load(id: String): LearningRate {
            return values().firstOrNull { it.id == id } ?: throw Exception("Learning rate ${id} not found")
        }
    }
}