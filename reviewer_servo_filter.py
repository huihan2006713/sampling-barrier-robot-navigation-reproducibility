"""One-step servo-aware supporting-plane filter for the declared lag plant.

The filter uses measured (p,v), known tau and acceleration cap. The radial
displacement of the saturated/exponential servo is minimized continuously
under the worst inward disturbance projection. Feasibility is checked at
each update; this is a conditional one-step certificate for the exact model,
not a differential-drive or uncertain-actuator guarantee.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import minimize

from simulate import D, R, TMAX, V, control
from reviewer_experiments import GOAL, OFFSETS, OUT, REGIMES, SEEDS, scenario, save, servo_hold


def radial_lower_bound(p, v, u, c, hold, tau, acceleration_limit=2.):
    """Minimum of n·(position(s)-c)-R for worst n·d=-D over 0..hold."""
    n = (p - c) / np.linalg.norm(p - c)
    h = float(np.linalg.norm(p - c) - R)
    error = u - v
    size = np.linalg.norm(error)
    if size < 1e-14:
        return min(h, h + (n @ v - D) * hold)
    direction = error / size
    t1 = min(hold, max(0., (size - acceleration_limit * tau) / acceleration_limit))
    an = acceleration_limit * float(n @ direction)
    vn = float(n @ v)
    candidates = [h]
    if t1 > 0:
        candidates.append(h + (vn - D) * t1 + .5 * an * t1*t1)
        if abs(an) > 1e-14:
            s = (D-vn)/an
            if 0 < s < t1:
                candidates.append(h + (vn-D)*s + .5*an*s*s)
    h += (vn-D)*t1 + .5*an*t1*t1
    v1 = v + acceleration_limit * direction * t1
    t2 = hold-t1
    if t2 > 0:
        en = float(n @ (u-v1))
        un = float(n @ u)
        def clearance(s):
            return h + (un-D)*s - tau*en*(-np.expm1(-s/tau))
        candidates.append(clearance(t2))
        if abs(en) > 1e-14:
            ratio = (un-D)/en
            if ratio > 0:
                s = -tau*math.log(ratio)
                if 0 < s < t2:
                    candidates.append(clearance(s))
    return min(candidates)


def viability_lower_bound(p, v, u, c, hold, tau):
    """Current hold and a conservative outward recovery with frozen normal."""
    n = (p-c)/np.linalg.norm(p-c)
    current = radial_lower_bound(p,v,u,c,hold,tau)
    p1,v1,_ = servo_hold(p.copy(),v.copy(),u,-D*n,hold,tau,c,2.)
    h1=float(n@(p1-c)-R)
    if h1 <= -1e-8:
        return min(current,h1)
    # Encode the frozen supporting plane as a synthetic circle center, so
    # radial_lower_bound uses the same normal and exactly h1 initial margin.
    surrogate_center=p1-(R+h1)*n
    backup=radial_lower_bound(p1,v1,V*n,surrogate_center,2.,tau)
    return min(current,backup)


def aware_command(p, v, raw, c, hold, tau):
    lower = viability_lower_bound(p,v,raw,c,hold,tau)
    if lower >= -1e-9:
        return raw, False, False
    n = (p-c)/np.linalg.norm(p-c)
    outward = V*n
    if viability_lower_bound(p,v,outward,c,hold,tau) < -1e-9:
        return outward, True, True
    fit = minimize(lambda u: float(np.sum((u-raw)**2)), outward,
                   method='SLSQP', constraints=[
                       {'type':'ineq','fun':lambda u: viability_lower_bound(p,v,u,c,hold,tau)},
                       {'type':'ineq','fun':lambda u: V*V-float(u@u)}],
                   options={'maxiter':40,'ftol':1e-9})
    if fit.success and np.linalg.norm(fit.x) <= V+1e-7 and viability_lower_bound(p,v,fit.x,c,hold,tau) >= -1e-8:
        return fit.x, True, False
    return outward, True, False


def run(method, offset, regime, seed, tau):
    p,bias=scenario(offset,seed)
    c=np.array([3.,offset]);v=np.zeros(2);t=0.;dt=.5
    worst=float(np.linalg.norm(p-c)-R);corrected=unavoidable=0
    worst_predicted=float('inf')
    status='timeout'
    while t < TMAX-1e-10:
        hold=min(dt,TMAX-t)
        raw,_=control(p,c,GOAL,hold,method)
        if raw is None:
            status='infeasible';break
        u,changed,impossible=aware_command(p,v,raw,c,hold,tau)
        corrected += int(changed)
        unavoidable += int(impossible)
        predicted=viability_lower_bound(p,v,u,c,hold,tau)
        worst_predicted=min(worst_predicted,predicted)
        if not impossible:
            assert predicted >= -1e-8, (method,offset,regime,seed,tau,t,predicted)
        n=(p-c)/np.linalg.norm(p-c)
        d=bias if regime=='bias' else -D*n
        p,v,minimum=servo_hold(p,v,u,d,hold,tau,c,2.)
        worst=min(worst,minimum);t+=hold
        if worst < -1e-8:
            status='violation';break
        if np.linalg.norm(p-GOAL)<=.15:
            status='arrived';break
    return dict(experiment='servo_aware',method=method,dt=dt,tau_v=tau,
                offset=offset,regime=regime,seed=seed,status=status,
                min_clearance=worst,time=t,corrected_updates=corrected,
                min_predicted_lower_bound=worst_predicted,
                uncertifiable_updates=unavoidable)


def main():
    rows=[run(method,offset,regime,seed,tau)
          for tau in (.1,.3) for method in ('SRCBF','CAP')
          for offset in OFFSETS for regime in REGIMES for seed in SEEDS]
    summary=save('servo_aware',rows,('tau_v','method'))
    (OUT/'servo_aware_protocol.json').write_text(json.dumps(dict(
        dt=.5,tau_v=[.1,.3],acceleration_limit=2.,
        state='True current position and servo velocity, exactly known parameters',
        constraint='Continuous minimum of current frozen supporting-plane clearance under n·d >= -D, plus a two-second full-speed outward recovery using the same plane and worst inward disturbance',
        fallback='Full-speed radial outward command if SLSQP cannot find a checked feasible command',
        uncertifiable='If even the outward command violates the one-step lower bound, this test counts an uncertifiable update and still simulates that command',
        limitation='One-step calculation and backup heuristic under exactly known servo parameters; no recursive feasibility proof, hardware, localization uncertainty or wheel dynamics'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('uncertifiable',sum(r['uncertifiable_updates'] for r in rows))


if __name__=='__main__':
    main()
