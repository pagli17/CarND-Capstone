import rospy

from lowpass import LowPassFilter
from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
STOP_BRAKE = 400.0

LOGGING_RATE = 1 # s 

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        kp = 0.7
        ki = 0.00007
        kd = 0.1
        mn = 0.  # Minimum throttle value
        mx = 0.2  # Maximum throttle value
        self.throttle_pid_controller = PID(kp, ki, kd, mn, mx)

        v_tau = 0.5  
        v_ts = .02 
        self.vel_lpf = LowPassFilter(v_tau, v_ts)
        
        s_tau = 0.5
        s_ts = .02
        self.steer_lpf = LowPassFilter(s_tau, s_ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()
        self.log_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        
        # Compute sample time
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        # DBW not enabled
        if not dbw_enabled:
            self.throttle_pid_controller.reset()
            # Log
            if (current_time - self.log_time) > LOGGING_RATE:
                rospy.logwarn("[CONTROLLER] enabled={}".format(dbw_enabled))
                self.log_time = current_time
            return 0.0, 0.0, 0.0
        
        # Steering controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        # steering = self.steer_lpf.filt(steering)

        # Compute linear velocity error
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel
        
        # Velocity controller
        throttle = self.throttle_pid_controller.step(vel_error, sample_time)
        throttle = self.vel_lpf.filt(throttle)
        
        brake = 0.0
        # Car stopped
        if linear_vel == 0.0 and current_vel < 0.1:
            throttle = 0.0
            brake = STOP_BRAKE
        # Car braking
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        # Log
        if (current_time - self.log_time) > LOGGING_RATE:
            rospy.logwarn("[CONTROLLER] current_lin_vel={:.2f}, target_lin_vel={:.2f}, target_ang_vel={:.2f}".format(current_vel, linear_vel, angular_vel))
            rospy.logwarn("[CONTROLLER] throttle={:.2f}, brake={:.2f}, steering={:.2f}".format(throttle, brake, steering))
            self.log_time = current_time

        return throttle, brake, steering
