"""Native PRO4000 thermal pacing around the unchanged frozen v2 task updates."""
import json
import os
from pathlib import Path
import sys
import time

POLICY={'profile':'pro4000-thermal-pacing-v1','minimum_update_start_interval_seconds':15,
    'cooling_trigger_c':65,'cooling_target_c':55,'emergency_c':82,
    'evaluation_check_every_windows':16,'poll_seconds':2,'maximum_cooling_wait_seconds':600,
    'driver_clock_power_or_tcc_changes':False,'optimizer_math_token_order_changes':False}

class Pacer:
    def __init__(self,query,stop,record,sleep=time.sleep,clock=time.monotonic,policy=None):
        self.query,self.stop,self.record,self.sleep,self.clock=query,stop,record,sleep,clock
        self.policy=dict(POLICY if policy is None else policy)
        self.previous_update=None
    def check(self,stage):
        start=self.clock();wait=0.0;cooling=False
        while True:
            if self.stop():raise RuntimeError('STOP marker during thermal pacing; preserve incomplete state')
            temperature=int(self.query())
            if temperature>=self.policy['emergency_c']:
                self.record({'stage':stage,'temperature_c':temperature,'event':'emergency_thermal_stop'})
                raise RuntimeError(f'Emergency thermal stop at {temperature} C; no automatic retry')
            cooling=cooling or temperature>self.policy['cooling_trigger_c']
            interval=(stage=='update' and self.previous_update is not None and
                self.clock()-self.previous_update<self.policy['minimum_update_start_interval_seconds'])
            if not interval and (not cooling or temperature<=self.policy['cooling_target_c']):break
            if self.clock()-start>=self.policy['maximum_cooling_wait_seconds']:
                raise RuntimeError('Thermal cooling timeout; no automatic retry')
            self.sleep(self.policy['poll_seconds']);wait+=self.policy['poll_seconds']
        if stage=='update':self.previous_update=self.clock()
        self.record({'stage':stage,'temperature_c':temperature,'wait_seconds':wait,'time':time.time()})
        return str(temperature)

