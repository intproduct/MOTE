from fitmotn.experiments.qwen_joint.thermal import Pacer

class Clock:
    def __init__(self):self.now=0
    def clock(self):return self.now
    def sleep(self,s):self.now+=s

def check(values,stop=False,previous=None):
    clock=Clock();iterator=iter(values);current=values[-1];log=[]
    def query():
        nonlocal current
        current=next(iterator,current)
        return current
    p=Pacer(query,lambda:stop,log.append,clock.sleep,clock.clock);p.previous_update=previous
    return p,clock,log

p,c,l=check([50]);assert p.check('update')=='50' and c.now==0
p,c,l=check([70,66,55]);assert p.check('update')=='55' and c.now==4
p,c,l=check([50],previous=0);assert p.check('update')=='50' and c.now>=15
p,c,l=check([82])
try:p.check('update');raise AssertionError('Emergency was masked')
except RuntimeError as e:assert 'Emergency' in str(e) and c.now==0
p,c,l=check([40],stop=True)
try:p.check('evaluation');raise AssertionError('STOP ignored')
except RuntimeError as e:assert 'STOP' in str(e)
p,c,l=check([70])
try:p.check('evaluation');raise AssertionError('Infinite cooldown')
except RuntimeError as e:assert 'timeout' in str(e) and c.now==600
print('6 thermal pacing checks passed; no GPU imports or use')
