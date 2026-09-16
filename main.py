"""Rock Paper Scissors vs AI — optimized real-time hand detection edition.

Fast pipeline:
- Camera frames are kept at display FPS while MediaPipe runs on a small frame.
- MediaPipe VIDEO mode keeps temporal tracking between frames.
- Gesture classification uses landmark angles + normalized distances, not just Y coordinates.
- A short confidence-weighted temporal smoother rejects unstable frames and resets between rounds.
- Low-light enhancement is applied only when the scene is actually dark.
"""
import json, os, random, time, urllib.request
from collections import Counter, deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import sounddevice as sd
    SOUND_AVAILABLE = True
except Exception:
    SOUND_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
STATS_PATH = os.path.join(SCRIPT_DIR, "stats.json")
WINDOW = "Rock Paper Scissors vs AI"
MOVES = ["Rock", "Paper", "Scissors"]
BEATS = {"Rock":"Scissors", "Scissors":"Paper", "Paper":"Rock"}

# Performance knobs. Lower DETECT_WIDTH = faster; 640 is a good quality/speed point.
DETECT_WIDTH = 640
DETECT_INTERVAL = 1.0 / 30.0
CAPTURE_WINDOW = 0.32
SMOOTHING_FRAMES = 6
MIN_STABLE_CONFIDENCE = 0.58

SAMPLE_RATE = 44100
SOUND_ENABLED = True

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def load_stats():
    default = {"wins":0,"losses":0,"draws":0,"matches_played":0,"matches_won":0,"best_streak":0}
    try:
        with open(STATS_PATH) as f:
            default.update(json.load(f))
    except Exception:
        pass
    return default

def save_stats(s):
    try:
        with open(STATS_PATH, "w") as f: json.dump(s, f)
    except Exception: pass

def tone(freq, duration, volume=.22):
    t=np.linspace(0,duration,int(SAMPLE_RATE*duration),False)
    w=volume*np.sin(2*np.pi*freq*t)
    fade=min(200,len(w)//4)
    if fade:
        e=np.ones(len(w)); e[:fade]=np.linspace(0,1,fade); e[-fade:]=np.linspace(1,0,fade); w*=e
    return w.astype(np.float32)
SOUNDS={"tick":tone(700,.07),"shoot":tone(1300,.15),"win":np.concatenate([tone(x,.08) for x in (523,659,784)]),"lose":np.concatenate([tone(x,.08) for x in (392,311,233)]),"draw":tone(440,.16)}
def play(name):
    if SOUND_AVAILABLE and SOUND_ENABLED:
        try: sd.play(SOUNDS[name], samplerate=SAMPLE_RATE)
        except Exception: pass

CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

def draw_hand(frame, lm, color=(80,220,120)):
    h,w=frame.shape[:2]; p=[(int(x.x*w),int(x.y*h)) for x in lm]
    for a,b in CONNECTIONS: cv2.line(frame,p[a],p[b],color,2,cv2.LINE_AA)
    for x,y in p: cv2.circle(frame,(x,y),3,(255,255,255),-1,cv2.LINE_AA)

def angle(a,b,c):
    # angle ABC in degrees
    ba=np.array([a.x-b.x,a.y-b.y,a.z-b.z],dtype=np.float32)
    bc=np.array([c.x-b.x,c.y-b.y,c.z-b.z],dtype=np.float32)
    den=np.linalg.norm(ba)*np.linalg.norm(bc)
    if den<1e-6:return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/den,-1,1))))

def dist(a,b):
    return float(np.linalg.norm(np.array([a.x-b.x,a.y-b.y,a.z-b.z],dtype=np.float32)))

def classify_hand(lm):
    """Return (gesture, confidence).

    Finger state is based primarily on joint angle and normalized fingertip distance,
    making it much less sensitive to camera rotation than tip.y < pip.y.
    """
    wrist=lm[0]
    fingers=((5,6,8),(9,10,12),(13,14,16),(17,18,20))
    states=[]; strengths=[]
    palm=max(dist(lm[0],lm[9]),1e-4)
    for mcp,pip,tip in fingers:
        a=angle(lm[mcp],lm[pip],lm[tip])
        reach=dist(lm[tip],wrist)/palm
        # Straight finger: joint angle near 180 and fingertip clearly away from wrist.
        s_angle=np.clip((a-135.0)/35.0,0,1)
        s_reach=np.clip((reach-1.25)/0.55,0,1)
        score=.68*s_angle+.32*s_reach
        states.append(score>.50)
        strengths.append(score if states[-1] else 1-score)
    i,m,r,p=states
    if sum(states)==0:
        gesture="Rock"
    elif i and m and not r and not p:
        gesture="Scissors"
    elif sum(states)==4:
        gesture="Paper"
    else:
        gesture=None
    if gesture is None:
        # Distance from the nearest valid RPS finger pattern.
        patterns={"Rock":(0,0,0,0),"Paper":(1,1,1,1),"Scissors":(1,1,0,0)}
        raw=np.array([1 if x else 0 for x in states])
        d={k:1-float(np.mean(np.abs(raw-np.array(v)))) for k,v in patterns.items()}
        best=max(d,key=d.get)
        return None, float(d[best])*.55
    return gesture, float(np.mean(strengths))

class GestureEngine:
    def __init__(self, landmarker):
        self.landmarker=landmarker
        self.last_detect=0.0
        self.timestamp=0
        self.raw=None
        self.raw_conf=0.0
        self.history=deque(maxlen=SMOOTHING_FRAMES)
        self.last_landmarks=None
        self.last_boost=False

    def process(self, frame):
        now=time.perf_counter()
        if now-self.last_detect < DETECT_INTERVAL and self.last_landmarks is not None:
            return self.raw, self.raw_conf, self.last_landmarks, self.last_boost
        self.last_detect=now
        small=frame
        h,w=frame.shape[:2]
        if w>DETECT_WIDTH:
            nw=DETECT_WIDTH; nh=int(h*nw/w); small=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA)
        boosted=False
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        if float(gray.mean())<78:
            # Cheap gamma-only boost; CLAHE is reserved for very dark frames.
            lut=np.array([min(255,int(255*((i/255.0)**(1/1.45)))) for i in range(256)],dtype=np.uint8)
            small=cv2.LUT(small,lut); boosted=True
            if gray.mean()<52:
                lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
                l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(6,6)).apply(l)
                small=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
        rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
        ts=int(time.time()*1000)
        if ts<=self.timestamp: ts=self.timestamp+1
        self.timestamp=ts
        result=self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),ts)
        self.last_landmarks=result.hand_landmarks[0] if result.hand_landmarks else None
        self.last_boost=boosted
        if self.last_landmarks:
            self.raw,self.raw_conf=classify_hand(self.last_landmarks)
            self.history.append((self.raw,self.raw_conf))
        else:
            self.raw,self.raw_conf=None,0.0
            self.history.append((None,0.0))
        return self.raw,self.raw_conf,self.last_landmarks,self.last_boost

    def stable(self):
        vals=[(g,c) for g,c in self.history if g]
        if not vals:return None,0.0
        scores={m:0.0 for m in MOVES}
        for g,c in vals:scores[g]+=max(c,.05)
        move=max(scores,key=scores.get); total=sum(scores.values()) or 1
        conf=scores[move]/total
        return (move,conf) if conf>=MIN_STABLE_CONFIDENCE else (None,conf)

def ai_choose(difficulty,history):
    if difficulty=="Hard" and len(history)>=3 and random.random()<.7:
        most=Counter(history).most_common(1)[0][0]
        return {"Rock":"Paper","Paper":"Scissors","Scissors":"Rock"}[most]
    return random.choice(MOVES)

def winner(p,a):
    if p==a:return "Draw"
    return "You Win!" if BEATS[p]==a else "AI Wins!"

def icon(frame,move,center,r=45,color=(255,255,255)):
    x,y=center
    if move=="Rock": cv2.circle(frame,(x,y),r,color,-1,cv2.LINE_AA)
    elif move=="Paper": cv2.rectangle(frame,(x-r,y-r),(x+r,y+r),color,-1)
    elif move=="Scissors":
        cv2.line(frame,(x-r//2,y-r//2),(x+r//2,y+r//2),color,8,cv2.LINE_AA); cv2.line(frame,(x-r//2,y+r//2),(x+r//2,y-r//2),color,8,cv2.LINE_AA)
    else: cv2.putText(frame,"?",(x-15,y+15),cv2.FONT_HERSHEY_SIMPLEX,1.3,color,3)

def box(frame,x1,y1,x2,y2,alpha=.58):
    o=frame.copy();cv2.rectangle(o,(x1,y1),(x2,y2),(18,20,26),-1);cv2.addWeighted(o,alpha,frame,1-alpha,0,frame)

def main():
    global SOUND_ENABLED
    ensure_model(); stats=load_stats()
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,60); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened(): print("Could not open webcam."); return
    options=mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,num_hands=1,
        min_hand_detection_confidence=.55,min_hand_presence_confidence=.50,min_tracking_confidence=.50)
    landmarker=mp_vision.HandLandmarker.create_from_options(options); engine=GestureEngine(landmarker)
    state="MENU"; best_of=3; difficulty="Easy"; match={"score":{"You":0,"AI":0},"history":[],"streak":0,"owner":None}
    countdown=0; capture_start=0.0; retry_time=0.0; result_time=0; capture=[]; player=ai=outcome=None; paused=False
    while True:
        ok,frame=cap.read()
        if not ok:break
        frame=cv2.flip(frame,1)
        gesture,conf,lm,boost=engine.process(frame)
        if lm: draw_hand(frame,lm)
        stable,stable_conf=engine.stable()
        h,w=frame.shape[:2]
        if boost: cv2.putText(frame,"LOW LIGHT BOOST",(w-190,28),0,.55,(0,230,255),1)
        cv2.putText(frame,f"Detection: {stable or '—'}  {stable_conf*100:.0f}%",(18,h-18),0,.55,(190,210,220),1)
        if paused:
            box(frame,0,0,w,h,.60);cv2.putText(frame,"PAUSED",(w//2-85,h//2),0,1.4,(255,255,255),3);cv2.putText(frame,"P to resume",(w//2-65,h//2+42),0,.65,(180,180,180),2)
        elif state=="MENU":
            cv2.putText(frame,"ROCK  PAPER  SCISSORS  vs AI",(30,50),0,.9,(255,255,255),2)
            cv2.putText(frame,f"Best of {best_of}   |   Difficulty: {difficulty}",(30,88),0,.68,(0,220,255),2)
            cv2.putText(frame,"1 / 3 / 5 / 7   choose match",(30,122),0,.58,(190,190,200),1)
            cv2.putText(frame,"E / D difficulty   |   T practice",(30,150),0,.58,(190,190,200),1)
            cv2.putText(frame,"SPACE  start match",(30,192),0,.72,(80,235,130),2)
            cv2.putText(frame,f"All-time  W {stats['wins']}  L {stats['losses']}  D {stats['draws']}  |  Best streak {stats['best_streak']}",(30,h-55),0,.52,(180,180,190),1)
        elif state=="PRACTICE":
            cv2.putText(frame,"PRACTICE  —  M to return",(25,42),0,.75,(255,255,255),2)
            icon(frame,stable,(w-90,105),55,(80,220,130) if stable else (100,100,110))
            cv2.putText(frame,stable or "Show one clear hand",(25,h-60),0,.72,(80,235,130) if stable else (240,90,90),2)
            cv2.rectangle(frame,(25,h-38),(245,h-22),(70,70,80),1);cv2.rectangle(frame,(25,h-38),(25+int(220*stable_conf),h-22),(80,220,130),-1)
        elif state=="WAITING":
            cv2.putText(frame,"SPACE  play round",(25,45),0,.78,(255,255,255),2)
            if stable: cv2.putText(frame,f"Ready: {stable}",(25,80),0,.68,(80,235,130),2)
        elif state=="COUNTDOWN":
            elapsed=time.time()-countdown; rem=3-int(elapsed); label=str(rem) if rem>0 else "SHOOT!"
            if label!=getattr(main,'last_label',None): play("shoot" if label=="SHOOT!" else "tick");main.last_label=label
            cv2.putText(frame,label,(w//2-75,h//2+25),0,1.8,(255,255,255),4)
            if elapsed>=3:
                state="CAPTURE"
                capture=[]
                capture_start=time.perf_counter()
                engine.reset_smoothing()
                countdown=0
        elif state=="CAPTURE":
            if stable: capture.append((stable,stable_conf))
            cv2.putText(frame,"HOLD YOUR GESTURE",(25,45),0,.78,(0,230,255),2)
            if time.perf_counter()-capture_start>=CAPTURE_WINDOW:
                if capture:
                    weighted={m:0 for m in MOVES}
                    for g,c in capture: weighted[g]+=max(c,.1)
                    player=max(weighted,key=weighted.get); ai=ai_choose(difficulty,match['history']); match['history'].append(player); outcome=winner(player,ai)
                    if outcome=="You Win!": match['score']['You']+=1;stats['wins']+=1;owner="You"
                    elif outcome=="AI Wins!": match['score']['AI']+=1;stats['losses']+=1;owner="AI"
                    else: stats['draws']+=1;owner=None
                    if owner==match['owner']:match['streak']+=1
                    elif owner:match['owner']=owner;match['streak']=1
                    else:match['owner']=None;match['streak']=0
                    stats['best_streak']=max(stats['best_streak'],match['streak']);save_stats(stats);result_time=time.time();state="RESULT"
                else: state="RETRY";retry_time=time.time()
        elif state=="RETRY":
            cv2.putText(frame,"Hand not clear — try again",(25,48),0,.8,(80,90,255),2)
            if time.time()-retry_time>1.0:
                state="COUNTDOWN"
                countdown=time.time()
                engine.reset_smoothing()
                main.last_label=None
        elif state=="RESULT":
            icon(frame,player,(w//2-110,115),50,(255,220,0));icon(frame,ai,(w//2+110,115),50,(0,180,255))
            cv2.putText(frame,player,(w//2-155,185),0,.65,(255,220,0),2);cv2.putText(frame,ai,(w//2+85,185),0,.65,(0,180,255),2)
            cv2.putText(frame,outcome,(w//2-100,250),0,1.0,(80,235,130) if outcome=='You Win!' else (255,255,255),3)
            if time.time()-result_time>1.8:
                needed=best_of//2+1
                if match['score']['You']>=needed or match['score']['AI']>=needed:
                    stats['matches_played']+=1
                    if match['score']['You']>=needed:stats['matches_won']+=1
                    save_stats(stats);state="MATCH_OVER"
                else:state="WAITING"
        elif state=="MATCH_OVER":
            win=match['score']['You']>match['score']['AI']
            cv2.putText(frame,"YOU WIN THE MATCH!" if win else "AI WINS THE MATCH",(w//2-180,110),0,1.0,(80,235,130) if win else (255,100,100),3)
            cv2.putText(frame,f"{match['score']['You']}  —  {match['score']['AI']}",(w//2-70,170),0,1.2,(255,255,255),3)
            cv2.putText(frame,"SPACE rematch   |   M menu",(w//2-125,225),0,.62,(190,190,200),2)
        try:
            cv2.imshow(WINDOW,frame)
            key=cv2.waitKey(1)&0xFF
        except cv2.error as exc:
            if "The function is not implemented" in str(exc) or "highgui" in str(exc).lower():
                print("\nOpenCV GUI support is unavailable.\n")
                print("Run setup_windows.bat to replace the headless/OpenCV 5 build with the desktop OpenCV build.")
                print("Then restart this program.\n")
                break
            raise
        if key==ord('q'):break
        if key==ord('p'):paused=not paused;continue
        if key==ord('s'):SOUND_ENABLED=not SOUND_ENABLED
        if key==ord('m'):
            state="MENU";match={"score":{"You":0,"AI":0},"history":[],"streak":0,"owner":None}
        if state=="MENU":
            if key in (ord('1'),ord('3'),ord('5'),ord('7')):best_of=int(chr(key))
            if key==ord('e'):difficulty="Easy"
            if key==ord('d'):difficulty="Hard"
            if key==ord('t'):state="PRACTICE"
            if key==32:
                match={"score":{"You":0,"AI":0},"history":[],"streak":0,"owner":None}
                engine.reset_smoothing()
                state="WAITING"
        elif state=="WAITING" and key==32:
            state="COUNTDOWN"
            countdown=time.time()
            engine.reset_smoothing()
            main.last_label=None
        elif state=="MATCH_OVER" and key==32:match={"score":{"You":0,"AI":0},"history":[],"streak":0,"owner":None};state="WAITING"
    cap.release();landmarker.close();cv2.destroyAllWindows()

if __name__=="__main__": main()
