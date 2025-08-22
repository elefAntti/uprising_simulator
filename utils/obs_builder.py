
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math, numpy as np

Vec2 = Tuple[float,float]; Pose = Tuple[Vec2,float]

def _rot(p: Vec2, ca: float, sa: float) -> Vec2: return (p[0]*ca - p[1]*sa, p[0]*sa + p[1]*ca)
def _sub(a: Vec2, b: Vec2) -> Vec2: return (a[0]-b[0], a[1]-b[1])
def _norm(a: Vec2) -> float: return math.hypot(a[0], a[1])

def _ray_to_rect_distance(p: Vec2, phi: float, S: float) -> float:
    x, y = p; dx, dy = math.cos(phi), math.sin(phi)
    ts = []
    if abs(dx)>1e-9: ts += [(0.0-x)/dx, (S-x)/dx]
    if abs(dy)>1e-9: ts += [(0.0-y)/dy, (S-y)/dy]
    best = float("inf")
    for t in ts:
        if t<=0: continue
        X, Y = x+t*dx, y+t*dy
        if -1e-6 <= X <= S+1e-6 and -1e-6 <= Y <= S+1e-6:
            best=min(best,t)
    return best if best<float('inf') else S*2.0

def _wall_features(world_pos: Vec2, ang: float, S: float, ray_deg):
    x,y = world_pos; base=[x/S,(S-x)/S,y/S,(S-y)/S]
    rays=[min(_ray_to_rect_distance(world_pos, ang+math.radians(a), S)/S,1.0) for a in ray_deg]
    side=min([("left",x),("right",S-x),("bottom",y),("top",S-y)], key=lambda t:t[1])[0]
    world_n={"left":(1,0),"right":(-1,0),"bottom":(0,1),"top":(0,-1)}[side]
    ca,sa=math.cos(-ang),math.sin(-ang); nx,ny=_rot(world_n,ca,sa)
    return np.array(base,dtype=np.float32), np.array(rays,dtype=np.float32), np.array([nx,ny],dtype=np.float32)

def _corners(S: float): return [(0.0,0.0),(0.0,S),(S,0.0),(S,S)]

def _nn_match(prev: List[Vec2], curr: List[Vec2], tol: float=0.35):
    mapping=[None]*len(curr); used=set()
    for j,p in enumerate(curr):
        best_i,best_d=None,float('inf')
        for i,q in enumerate(prev):
            if i in used: continue
            d= _norm(_sub(p,q))
            if d<best_d: best_d, best_i=d, i
        if best_i is not None and best_d<=tol: mapping[j]=best_i; used.add(best_i)
    return mapping

@dataclass
class EgoObsBuilder:
    field_size: float = 1.5; k_red: int = 8; k_green: int = 4
    ray_deg: tuple = (-90,-60,-30,0,30,60,90); default_dt: float = 0.05
    prev_time: Optional[float]=None; prev_red: Optional[List[Vec2]]=None; prev_green: Optional[List[Vec2]]=None

    def reset(self): self.prev_time=None; self.prev_red=None; self.prev_green=None

    def build(self, agent_idx:int, bot_coords:List[Pose], red_coords:List[Vec2], green_coords:List[Vec2],
              base_own:Vec2, base_opp:Vec2, time_now:Optional[float]=None, red_vel=None, green_vel=None, side_id:float=+1.0)->Dict[str,np.ndarray]:
        S=self.field_size
        dt=self.default_dt if (time_now is None or self.prev_time is None) else max(1e-3, min(0.25, time_now-self.prev_time))
        self.prev_time = time_now if time_now is not None else (self.prev_time or 0.0)+dt
        (agent_pos, agent_ang)=bot_coords[agent_idx]; ca,sa=math.cos(-agent_ang),math.sin(-agent_ang)
        def to_ego(p:Vec2)->Vec2: return _rot(_sub(p,agent_pos), ca, sa)
        wall_base, wall_rays, wall_normal = _wall_features(agent_pos, agent_ang, S, self.ray_deg)
        own_vec = np.array(to_ego(base_own), dtype=np.float32)/S; opp_vec = np.array(to_ego(base_opp), dtype=np.float32)/S
        corners = np.array([to_ego(c) for c in _corners(S)], dtype=np.float32)/S
        team_base=(agent_idx//2)*2; teammate_idx=team_base+(1-(agent_idx%2)); opp_idxs=[i for i in range(len(bot_coords)) if (i//2)!=(agent_idx//2)]
        def pack_robot(i):
            (p,ang)=bot_coords[i]; rel=to_ego(p); yaw=ang-agent_ang; s,c=math.sin(yaw),math.cos(yaw)
            return np.array([rel[0]/S, rel[1]/S, c, s], dtype=np.float32)
        teammate=pack_robot(teammate_idx); opps=np.stack([pack_robot(i) for i in opp_idxs], axis=0) if opp_idxs else np.zeros((0,4),dtype=np.float32)

        def estimate_vel(prev, curr, provided):
            if provided is not None and len(provided)==len(curr): return [tuple(v) for v in provided]
            if prev is None or dt<=0.0 or not prev: return [(0.0,0.0) for _ in curr]
            map_prev=_nn_match(prev, curr, tol=0.35); vel=[]
            for j,p in enumerate(curr):
                i=map_prev[j]
                if i is None: vel.append((0.0,0.0))
                else:
                    q=prev[i]; vel.append(((p[0]-q[0])/dt, (p[1]-q[1])/dt))
            return vel

        red_v=estimate_vel(self.prev_red, red_coords, red_vel); green_v=estimate_vel(self.prev_green, green_coords, green_vel)
        self.prev_red=list(red_coords); self.prev_green=list(green_coords)

        def pack_balls(coords, vels, k):
            items=[]
            for p,v in zip(coords, vels):
                pe=to_ego(p); ve=_rot(v, ca, sa); d_opp=_norm(_sub(p, base_opp))/S; d_own=_norm(_sub(p, base_own))/S
                cg=[c for c in _corners(S) if _norm(_sub(base_opp,c))>1e-6]; d_corner=min(_norm(_sub(p,c)) for c in cg) if cg else S
                items.append((_norm(pe), pe, ve, d_opp, d_own, d_corner/S))
            items.sort(key=lambda t:t[0]); feats=[]; mask=[]
            for _,pe,ve,d_opp,d_own,d_corner in items[:k]:
                feats.append([pe[0]/S, pe[1]/S, ve[0], ve[1], d_opp, d_own, d_corner]); mask.append(1.0)
            while len(feats)<k: feats.append([0.0]*7); mask.append(0.0)
            return np.array(feats,dtype=np.float32), np.array(mask,dtype=np.float32)

        balls_red, mask_red = pack_balls(red_coords, red_v, self.k_red)
        balls_green, mask_green = pack_balls(green_coords, green_v, self.k_green)
        scalars = np.array([side_id], dtype=np.float32)

        flat = np.concatenate([scalars, wall_base, wall_rays, wall_normal, own_vec, opp_vec, corners.reshape(-1),
                               teammate, opps.reshape(-1) if opps.size else np.zeros(0,dtype=np.float32),
                               balls_red.reshape(-1), mask_red, balls_green.reshape(-1), mask_green], axis=0)
        return {"flat": flat}
