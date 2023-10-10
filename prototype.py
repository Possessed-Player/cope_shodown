import cv2 as cv
import numpy as np
import math



class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point({0}, {1})".format(self.x, self.y)

class Size:
    def __init__(self, w, h):
        self.w = w
        self.h = h


class ColorRange:
    def __init__(self, l, u):
        self.lower = np.array(l)
        self.upper = np.array(u)


class Platform:
    WIDTH = 31

    def __init__(self, lth, lvl, lcn, main=False):
        self.length   = lth
        self.level    = lvl
        self.location = Point(lcn[0], lcn[1]) 
        self.main     = main
    
    def get_edges(self):
        ledg = self.location
        redg = Point(self.location.x + self.length,
                     self.location.y)
        return ledg, redg

    def __repr__(self):
        return ("Platfrom -> level: {0}, x: {1}, y: {2}"
                .format(self.level, self.location.x, self.location.y))


class Direction:
    L = False
    R = True


class VDirection:
    U = False
    D = True


class State:
    STAND = 0
    JUMP  = 1
    DEAD  = 2
    FLY   = 3


class Obj:
    def __init__(self, p, s, c):
        self.position = Point(     p[0], p[1])
        self.size     = Size(      s[0], s[1]) 
        self.clr_rng  = ColorRange(c[0], c[1])




class Entity(Obj):
    def __init__(self, d, st, sp, p, sz, cr):
        super().__init__(p, sz, cr)
        self.direction = d
        self.state     = st
        self.speed     = sp

    def update_status(self, img, gs):
        raise NotImplementedError(" This method is not implemented ")


class Samurai(Entity):
    SPEED = 30
    SIZE  = (100, 100)
    CR    = ((75 , 130, 170),
             (120, 200, 255))

    def __init__(self):
        super().__init__(Direction.L, State.STAND, self.SPEED, 
                         (453, 354), self.SIZE, self.CR)

                
    def update_status(self, img, gs):
        # None -> Point
        # Find sword in image
        def find_sword():
            sc = (178, 187, 190)
            bsx = self.position.x - 40
            bsy = self.position.y - 40

            im = img[bsy:bsy+80, bsx:bsx+80].copy()
            res = np.where((im == sc).all(axis=2))
            
            if len(res[0]) == 0:
                return None
            else:
                return Point(bsx + res[1][0], 
                             bsy + res[0][0])


        swordp = find_sword()

        if swordp is None:
            self.direction = None
        elif swordp.x < self.position.x:
            self.direction = Direction.R
        elif swordp.x > self.position.x:
            self.direction = Direction.L
        else:
            self.direction = None

        pltfrm, _ = gs.map.get_platform(self.position) 

        if pltfrm is None:
            self.state = State.JUMP
        else:
            self.state = State.STAND



class Enemy(Entity):
    def __init__(self, d, st, sp, p, sz, cr):
        super().__init__(d, st, sp, p, sz, cr)

    def update_status(self, _, gs):
        old_enms = gs.old_enms[0]
        if old_enms is None: return
        # tuple of Enemy -> Enemy|None 
        # Return the closest Point(Enemy) from tuple of Enemy(s)
        #   relative to current object position, None if no Point is close
        #   acc is used as an accumulator for comparison
        def closest(tops):
            # rsf is a result so far accumulator 
            # Either None or (Enemy, int)
            # None -> if no close enemy found
            # (Enemy, int) -> accumulates the distance and the
            # that is considered the best match so far
            # Returns only Enemy once it reaches end of Enemy tuple
            def aux(tops, rsf=None):
                if tops == ():
                    return rsf[0] if not (rsf is None) else rsf
                else:
                    st = tops[0]
                    dst = math.dist((st.position.x, st.position.y), 
                                    (self.position.x, self.position.y))
                    if (type(self) == type(st) and 
                        (rsf is None or dst < (rsf[1]))):
                        return aux(tops[1:], (st, dst)) 
                    else: 
                        return aux(tops[1:], rsf)

            return aux(tops)
        
        
        oenm = closest(old_enms)

        if oenm is None: 
            return

        if oenm.position.x > self.position.x:
            self.direction = Direction.L
        elif oenm.position.x == self.position.x:
            self.direction = oenm.direction 
        else:
            self.direction = Direction.R

        if oenm.speed == 0:
            self.speed = self.DFSP
        else:
            self.speed = ((oenm.speed + 
                          abs(oenm.position.x - self.position.x)) // 2)

        return oenm 


class Bird(Enemy):
    CR = ((80 , 115, 140),
          (255, 120, 255))
    SIZE = (80, 80)
    DFSP = 5

    def __init__(self, *args):
        super().__init__(*args, sz=self.SIZE, cr=self.CR)
        self.state = State.FLY
 

class Burger(Enemy):
    CR = ((15, 120, 164),
          (20, 140, 240))
    SIZE = (80, 70)
    DFSP = 30

    def __init__(self, *args):
        super().__init__(*args, sz=self.SIZE, cr=self.CR)
        self.state = State.STAND 

    def update_status(self, imgs, gs):
        oenm = super().update_status(imgs, gs)

        if oenm is None: return

        pltfrm, _ = gs.map.get_platform(self.position) 

        if pltfrm is None:
            mid_pm = gs.map.pms[3]
            LOW_Y = 540
            if ((mid_pm.location.x < self.position.x < 
                    (mid_pm.location.x + mid_pm.length)) or
                    self.position.y > LOW_Y):
                self.state = State.DEAD
            else:
                self.state = State.JUMP 
        else:
            self.state = State.STAND



class GoldBag(Obj):
    CR = ((20 , 125, 255),
          (128, 218, 255))
    SIZE = (50, 40)

    def __init__(self):
        super().__init__((0, 0), self.SIZE, self.CR)
        self.count    = 15

		

class GameState:
    def __init__(self, lvl):
        self.running  = True
        self.samurai  = Samurai()
        self.lifes    = 9
        self.enemies  = () 
        self.old_enms = (None,) * 8
        self.gb       = GoldBag()
        self.map      = lvl 



class Action:
    def __init__(self, drc):
        self.direction = drc

class Move(Action):
    def __init__(self, drc, goal):
        super().__init__(drc)
        self.goal = goal 

class Attack(Action):
    def __init__(self, drc, dwn=False):
        super().__init__(drc)
        self.down_attk = dwn

class Jump(Move):
    def __init__(self, drc, goal, vdrc):
        super().__init__(drc, goal)
        self.vert_direction = vdrc


class Map:
    def __init__(self, *args, level=0):
        self.pms   = {}
        self.graph = {}
        for i, pm in enumerate(args):
            self.pms[i] = pm

        if level == 1:
            self.graph = {
                0:(2,3,1),
                1:(3,),
                2:(0,3,1,4),
                3:(0,1,4,6,7),
                4:(2,3,5),
                5:(4,3,6,7),
                6:(5,3,7),
                7:(3,)
                    }



    # int, int -> tuple of tuple of int
    # Return a set of sequences that
    # lead from start(platform)
    # to goal(platform)
    def paths(self, start, goal):
        def aux(rsf=(start,), visited=(), acc=()):
            if rsf == ():
                return acc
            if rsf[-1] == goal:
                return aux(rsf[:-1], visited, acc + (rsf,)) 
            elif len(rsf) > 6 or rsf[-1] in rsf[:-1]:
                return aux(rsf[:-1], visited + (rsf,), acc) 
            else:
                for node in self.graph[rsf[-1]]:
                    if (rsf + (node,)) in visited:
                        continue
                    else:
                        seq = rsf + (node,)
                        return aux(seq, visited + (seq,), acc)
                else:
                    return aux(rsf[:-1], visited, acc)


        lst = list(aux())

        lst.sort(key=len)
        tpl = tuple(lst)

        ntpl = ()
        for seq in tpl:
            nseq = ()
            for i in seq:
                nseq += (self.pms[i],)
            ntpl += (nseq,)
                

        return ntpl 


    # None -> tuple of Platfrom
    # Return all platforms in self
    # in a tuple
    def get_pms_astpl(self):
        return tuple(self.pms.values())

    # Point -> Platform, int|None, None
    # Find the platform and its' id 
    # if pnt lies on
    # if pnt isn't on any platform
    # return None, None
    def get_platform(self, pnt, hy=False):
        def find_platform(pms, p):
            if pms == ():
                return None, None
            else:
                pm = pms[0]
                if (pm.location.x - 10) <= p.x <= (pm.location.x + pm.length + 10):
                    if 10 < (pm.location.y - p.y) < 75:
                        return pm, abs(len(pms) - 8)

                return find_platform(pms[1:], p)
        
        return find_platform(self.get_pms_astpl(), pnt)



# CONSTANTS
GC_SP   = 470, 170
WIDTH   = 965
HEIGHT  = 600
MAP_1   = Map(Platform(84 , 2, (0  , 266)), Platform(117, 0, (0  , 533)),
              Platform(175, 3, (178, 133)), 
              Platform(606, 1, (178, 398),  main=True), 
		      Platform(142, 2, (411, 266)), Platform(175, 3, (611, 133)),
		      Platform(84 , 2, (879, 266)), Platform(118, 0, (845, 533)), 
              level=1) 
# ---------------------------------------------------------------------------------------
# DATA DEFINITIONS:
#
# Point is Point(int[0, WIDTH], int[0, HEIGHT])
# Interp. Represents a position/pixel on the game image/map/screen
#           int -> x coordinate
#           int -> y coordinate
#
PNT1 = Point(0, 0)                      # TOP-LEFT    Corner
PNT2 = Point(WIDTH // 2, HEIGHT // 2)   # MIDDLE
PNT2 = Point(WIDTH, HEIGHT)             # BOTTOM-LEFT Corner
#
# def fn_for_pt(pt : Point):
#   ...(pt.x, pt.y)
#
# Template rules used:
# - Atomic Non-Distinct : int
# - Atomic Non-Distinct : int
#
# Size is Size(int[1, 70], int[1, 70])
# Interp. Represent size of object/entity
#           int -> Width
#           int -> Height
#
S1 = Size(1, 1)     # Smallest possible size
S2 = Size(50, 50)   # Usual size
S3 = Size(70, 70)   # Largest possible size
#
# def fn_for_sz(s : Size):
#   ...(s.w, s.h)
#
# Template ruels used:
# - Atomic Non-Distinct : int
# - Atomic Non-Distinct : int
#
#
# INVARIANTS:
# - Only one Platform can be the main per game map.
#
# Platform is Platform(int, int, tuple(int, int)) 
# Interp. represent in game platforms player can walk/jump on
#			int 			-> Length of the platform
#			int 			-> Level of the platform from the ground 
#								with 0 being the lowest
#			tuple(int, int) -> Position of the top-left corner
#								of the platform
#			boolean			-> Whether the specified platform is the main platform
#								which means it contains level door in the middle
#
P1 = Platform(100, 2, (50, 100))
#
# def fn_for_plfrm(p : Platform):
# 	 ...(p.length,								# int 
# 		 p.level,								# int 
# 	 	 fn_for_pt(p.location),         		# Point 
# 		 p.main)								# boolean	
#
# Template rules used:
# - compound: 4 fields
# - Atomic Non-Distinct : int
# - Atomic Non-Distinct : int
# - Reference : Point
# - Atomic Non-Distinct : boolean
#
#
# Direction is one of:
# - Direction.L -> False
# - Direction.R -> True
# Interp. Represent horizontal direction 
#           L -> Left  direction
#           R -> Right direction
#
# <Examples are redundant in enumeration>
#
# def fnc_for_dr(dr : Direction):
# 	 if dr == Direction.L:
# 	 	 ...
# 	 elif dr == Direction.R:
# 	 	 ...
#
# Template rules used:
# - One of: 2 cases
# - Atomic distinct : False
# - Atomic distinct : True
#
#
# VDirection is one of:
# - Direction.U -> False
# - Direction.D -> True
# Interp. Represent vertical direction
#           U -> Up   direction
#           D -> Down direction
#
# <Exapmles are redundant in enumeration>
#
#
# State is one of:
# - State.STAND -> 0
# - State.JUMP  -> 1
# - State.DEAD  -> 2
# Interp. Represents the state of an entity
#           STAND -> Entity is standing
#           JUMP  -> Entity is in mid jump
#           DEAD  -> Entity is dead
# 
# <Examples are redundant in enumeration>
#
# def fnc_for_st(st : State):
# 	 if st == State.STAND:
# 	 	 ...
# 	 elif st == State.JUMP:
# 		 ...
# 	 elif st == State.DEAD:
#		 ...
#
# Template rules used:
# - One of: 3 cases
# - Atomic distinct : 0
# - Atomic distinct : 1
# - Atomic distinct : 2
#
#
# Obj is Obj(Point, Size, ColorRange)
# Interp. Represents a any game object moving or non-moving
#           Point      -> is object's position in game map
#           Size       -> is object's size
#           ColorRange -> is object's color 
#
OBJ1 = Obj((120, 40), (50, 50), ((0  , 0  , 0  ),
                                 (255, 255, 255)))
OBJ2 = Obj((70 , 20), (25, 35), ((120, 120, 120),
                                 (140, 130, 205)))
#
#
#
# Entity is Entity(Direction, State, int, tuple(int, int), tuple(int, int),
#                  tuple(tuple(int, int, int), tuple(int, int, int)))
# Interp. Represents a moving game object/entity
#           Direction  -> What direction is the entity facing
#           State      -> Current state of the entity
#           int        -> Entity movement speed
#           tuple      -> Position of the entity in game map
#           tuple      -> Size of the entity
#           tuple      -> Color of the entity in game
#           
#
E1 = Entity(Direction.L, State.STAND, 10, (50 ,  50), (30, 50), ((130, 70 , 10),
                                                                 (200, 120, 20))) 
E2 = Entity(Direction.R, State.JUMP,  19, (207, 403), (45, 60), ((30 , 40,  7),
                                                                 (100, 60, 13))) 
#
# 
#
# INVARIANT:
# A) There is only one samurai per game
# B) Samurai has 9 lifes per full run
#
# Samurai is Samurai() which is each of:
# - Entity
# - State
# Interp. Is the main playable character in the game
#           Entity -> represents the samurai as a game entity
#           State  -> current state of the samurai in game
#
#
#
# Enemy is Enemy(<Same as Entity>) (Inherits Entity)
# Interp. Represents an enemy in the game
#
E1 = Enemy((40 ,  20), Direction.L, 10, (40, 40), (70, 50), ((200, 200, 200),
                                                             (255, 255, 255)))
E2 = Enemy((300, 130), Direction.R, 15, (50, 45), (40, 40), ((40, 40, 40),
                                                             (75, 75, 75)))
#
#
#
# GameState is GameState(Level) and each of:
# - samurai
# - lives
# - tuple(enemies)
# - gb
# Interp. Represent the current game state where:
#           samurai -> is current game Samurai
#           lives   -> lives left in current game
#           enemies -> tuple of visible enemies
#           gb      -> is current game GoldBag
#
GS1 = GameState(MAP_1)
#
# def fn_for_gs(gs : GameState):
#     ...(fn_for_sm(gs.samurai),
#         gs.lives,
#         fn_for_loenm(gs.enemies),
#         fn_for_gb(gs.gb),
#         fn_for_lopltfrm(gs.level))
#
# Template rules used:
# - compound : 5 fields
# - Reference : Samurai
# - Atomic Non-Distinct : int
# - Tuple of : Reference : Enemy
# - Reference : GoldBag
# - Tuple of : Reference : Platform
#
#
#
# Generative Recursion Template:
#
# def gen_rec(x):
#   if isTrivial(x):
#       return trivialAnswer(x)
#   else:
#       return gen_rec(next_problem(x))
#
# Backtracking Search Template:
#
# def fn_for_x(x):
#   return fn_for_lox(x.children)
#
# def fn_for_lox(lox):
#   if empty(lox):
#       return False
#   elif isAnswer(lox[0]):
#       return return lox[0] 
#   else:
#       ans = fn_for_x(lox[0])
#       if ans is False:
#           return fn_for_lox(lox[1:])
#       else:
#           return ans
#
#
#
# Action is Action(Direction)
# Interp. Represents a Samurai action in game
#           Direction -> Is horizontal direction(left|right) of the action
#
A1 = Action(Direction.L)
A2 = Action(Direction.R)
# 
#
# Move is Move(Direction, int) (Inherits Action)
# Interp. Represents horizontal movement of the Samurai
#           Direction -> <Same as Action>
#           int       -> Is number of pixels to travel in the
#                        specified direction
#
M1 = Move(Direction.L, 15) # Move to the left  15 pixels
M2 = Move(Direction.R, 70) # Move to the right 70 pixels
#
#
# Attack is Attack(Direction, bool) or
#           Attack(Direction) Default bool is False - (Inherits Action)
# Interp. Represents attack by Samurai at specific direction
#           Direction -> <Same as Action>
#           bool      -> Is whether the attack is downwards or not
#
ATT1 = Attack(Direction.L       ) # Attack at the left     direction
ATT2 = Attack(Direction.R,  True) # Attack at the downward direction
#
#
# Jump is Jump(Direction, int, VDirection) (Inherits Move)
# Interp. Represent jump by Samurai at specific 2d direction
#           Direction  -> Horizontal vector
#           int        -> <Same as Move>
#           VDirection -> Vertical vector
#
J1 = Jump(Direction.L, 30, VDirection.U) # Jump Left-Up    30 pixels
J2 = Jump(Direction.R, 20, VDirection.D) # Jump Right-Down 20 pixels
#
#
# Path is tuple of Point
# Interp. Represents multiple points that lie on platforms
#           that lead to an objective at the end
#
PTH1 = (Point(50, 50), Point(100, 70), Point(300, 100))
#
#
# ---------------------------------------------------------------------------------------
# FUNCTIONS DEFINITIONS:
#
# -- WORLD DESIGN --
#
# Main Loop
#
# <Tested Implicitly>
#
def main():
    paused = False
    f_gen  = frame_gen("demo.avi")
    frm    = next(f_gen)
    gs     = GameState(MAP_1)
    path   = ()
    acn    = None
    while not (frm is None):
        if not paused:
            # Collect current(fresh) game data
            gs   = detect(frm, gs)
            # Based on GameState setup a set of optimal interventions 
            path = intervene(gs, path)
            # Decide based on GameState and current optimal Path
            # what to do next move, attack, jump or drop
            acn = best_action(gs, path, acn)
            # Draw both collected data and interventions on game img
            draw(frm, gs, path, acn) 
            # Get a new frame
            frm = next(f_gen)

        key = cv.waitKey(1)
        # Exit  if 'q' key is pressed,
        # Pause if 'p' key is pressed
        if key == ord('q'):
            break 
        elif key == ord('p'):
            paused = not paused
    

# Point, Point, int -> bool
# Whether point a is approximately
# equal to point b in term of coordinates
# in range of err
def approx_equal(a, b, err):
    return (abs(a.x - b.x) < err and 
            abs(a.y - b.y) < err)


# GameState, Path -> Path 
# Setup a new of optimal path(if Samurai is already in end goal
#   or op is empty or Samurai diverged away from op) 
#   to follow else reduce old path if Samurai
#   reached a sub-route
#       Path      -> Old path generated by this function
#       GameState -> Current state of the game
#
# <Use 'python tests.py actions' to test this function>
#
def intervene(gs, op):
    if gs.samurai is None or gs.gb is None:
        return () 

    if gs.samurai.state == State.JUMP:
        return op

    _, spi  = gs.map.get_platform(gs.samurai.position)
    _, gbpi = gs.map.get_platform(gs.gb.position)


    sms = gs.samurai.size



    # None -> None|int[0-6] 
    # Return None if Samurai is not following one of the tasks in op
    # else return the task index in path as int
    def sam_stask():
        for i, p in enumerate(op):
            _, pmi = gs.map.get_platform(p)
            if pmi == spi:
                return i 
            else:
                continue
        return None 


    # None -> Path
    # Return the path of least resistance
    # out of all possible paths
    # TODO: correctly find path of least resistance
    def optimal_path():
        # Point, Point, Point -> 0|1|2
        # Return 0 if a is closer to c than b
        #        1 if b is closer to c than b
        #        2 if a and b are at the same distance from c
        def closest_pnt(a, b, c):
            dsta = math.dist((a.x, a.y), (c.x, c.y))
            dstb = math.dist((b.x, b.y), (c.x, c.y))

            if dsta < dstb:
                return 0
            if dstb < dsta:
                return 1
            else:
                return 2

        # tuple of Platform -> Path
        # Convert platforms to Point(s)
        # and insert mandatory Point(s)
        def pms_topnts(topms):
            err = 40
            # Platform, tuple of Platform, Accumulator -> Path
            def aux(fst, topm, acc):
                if topm == ():
                    gbpnt = Point(gs.gb.position.x, acc[-1].y)
                    return (acc if approx_equal(gbpnt, acc[-1], err) else
                            acc + (gbpnt,))
                else:
                    snd = topm[0]
                    ledg, redg = snd.get_edges()
                    fledg, fredg = fst.get_edges()
                    if fst.level < snd.level:
                        # Ascending
                        cp = closest_pnt(ledg, redg, acc[-1]) 
                        y  = fst.location.y - (sms.h // 2)
                        ly = snd.location.y - (sms.h // 2)
                        if cp == 0:
                            x  = ledg.x - sms.w
                            lx = ledg.x + (sms.w // 2)
                        elif cp == 1:
                            x  = redg.x + sms.w
                            lx = redg.x - (sms.w // 2)
                        else:
                            x  = ledg.x - sms.w
                            lx = ledg.x + (sms.w // 2)
                        
                        close_pnt = Point( x,  y)
                        landg_pnt = Point(lx, ly)
                    else:
                        # Descending
                        y  = fst.location.y - (sms.h // 2)
                        ly = snd.location.y - (sms.h // 2)
                        if (ledg.x < (fledg.x - sms.w) < redg.x or 
                            ledg.x < (fledg.x - (sms.w // 4)) < redg.x):
                            x  = fledg.x
                            if ledg.x < fledg.x < redg.x:
                                lx = fledg.x - (sms.w // 4) 
                            else:
                                lx = redg.x - (sms.w // 2)
                        else:
                            x  = fredg.x
                            if ledg.x < fredg.x < redg.x:
                                lx = fredg.x + (sms.w // 4)
                            else:
                                lx = ledg.x + (sms.w // 2)

                        close_pnt = Point( x,  y)
                        landg_pnt = Point(lx, ly)

                    if approx_equal(close_pnt, acc[-1], err):
                        return aux(snd, topm[1:], acc + (landg_pnt,))
                    else:
                        return aux(snd, topm[1:], acc + (close_pnt, landg_pnt))

            return aux(topms[0], topms[1:], (gs.samurai.position,))
        

        return pms_topnts(gs.map.paths(spi, gbpi)[0])
    
    sst = sam_stask()
    if op == () or sst is None:
        path = optimal_path()
    else:
        path = op[sst:]

    return path


# GameState, Path, Action -> Action|None 
# Decide what is the best action to take based on given
# gs and pth to either move, attack, jump or drop
def best_action(gs, pth, oacn):
    if gs.samurai is None:
        return None

    # None -> Attack|Move|None
    # return Attack if target is within attack range
    # Move if target is coming from above
    # to move away from target
    # else None
    def should_attack_orflee():
        att_dst = 120
        lh_diff  = 25
        uh_diff  = 50
        spos = gs.samurai.position
        # Point -> Attack|Move|None
        def direction(epos):
            x_diff = epos.x - gs.samurai.position.x
            y_diff = epos.y - gs.samurai.position.y
            
            if x_diff > 0: 
                drc = Direction.R
            elif x_diff < 0: 
                drc = Direction.L
            else:
                drc = None

            if y_diff > lh_diff:
                return Attack(None, True)
            elif y_diff < -uh_diff: 
                if drc == Direction.R:
                    return Move(Direction.L, att_dst)
                else:
                    return Move(Direction.R, att_dst)
            elif drc is None:
                return None
            else:
                return Attack(drc)



        for enm in gs.enemies:
            if enm.state == State.DEAD:
                continue
            edst = math.dist((enm.position.x, enm.position.y),
                             (spos.x, spos.y))
            if edst <= att_dst:
                return direction(enm.position)
        else:
            return None


    # None -> Move
    # produce a move action based on the next path task
    def move():
        # Point, int, int -> Jump|None
        # Either jump or drop based on wether
        # level difference between pmi_a and pmi_b
        def jump(p, pmi_a, pmi_b):
            alvl = gs.map.pms[pmi_a].level
            blvl = gs.map.pms[pmi_b].level
            x_diff = p.x - gs.samurai.position.x

            if x_diff < 0:
                drc = Direction.L
            elif x_diff > 0:
                drc = Direction.R
            else:
                return None

            if alvl < blvl:
                return Jump(drc, abs(x_diff), VDirection.U)
            elif alvl > blvl:
                return Jump(drc, abs(x_diff), VDirection.D) 
            else:
                return None

        # Point -> Move|None
        # Calculate distance to point and return a Move action
        # based on that
        def move(p):
            x_diff = p.x - gs.samurai.position.x
            if x_diff < 0:
                return Move(Direction.L, abs(x_diff))
            elif x_diff > 0:
                return Move(Direction.R, abs(x_diff))
            else:
                return None

        _, ipm_i = gs.map.get_platform(pth[0])
        for i, pnt in enumerate(pth[1:]):
            _, pm_i = gs.map.get_platform(pnt)
            n       = i + 1
            npm_i = pm_i

            if pm_i != ipm_i:
                break

        if approx_equal(gs.samurai.position, pth[n-1], 50):
            return jump(pth[n], ipm_i, npm_i)
        else:
            return move(pth[n]) 

            
        



    if gs.enemies is None:
        attack = None
    else:
        attack = should_attack_orflee()

    if attack:
        return attack
    elif pth == ():
        return None
    elif len(pth) > 1 and gs.samurai.state == State.STAND:
        mv = move()
        if mv: return mv

    return oacn


# String -> Generator -> BGR Image, None
# Given video path return a generator that returns
#   the next frame as BGR Image or None when video ends
#
# <Use'python tests.py' to test this function>
#
def frame_gen(path : str):
    stream = cv.VideoCapture(path)

    while stream.isOpened():
        ret, frame = stream.read()
        if ret:
            yield frame[GC_SP[1]:(GC_SP[1] + HEIGHT),
                        GC_SP[0]:(GC_SP[0] + WIDTH)].copy()
        else:
            yield None 

# BGR Image -> None
# Draw given BGR Image and show it up in a window
#   Exit if 'q' key is pressed
#
# <Tested Implicitly>
#
def show_img(img):
    cv.imshow("Game", img)

# BGR Image, GameState, Path, Action|None -> None
#   Draw GameState and Path on given image and show it in a window
#
# <Tested Implicitly>
#
def draw(img, gs, pth, acn):
    if not gs.running:
        return

    TXT_LOC = (250, 450)
    FONT = cv.FONT_HERSHEY_SIMPLEX

    # Obj -> None
    # Draw red rectangles around given Obj on local img variable
    def draw_rect(obj):
        # cv.rectangle(image, top-left, bottom-right, color, thickness)
        if isinstance(obj, Obj):
            x = obj.position.x
            y = obj.position.y
            mx = obj.size.w // 2
            my = obj.size.h // 2
        else:
            x  = obj.x
            y  = obj.y
            mx = 50
            my = 50

        cv.rectangle(img, (x - mx, y - my),
                          (x + mx, y + my),
                     (0, 0, 255), 3)

    # Samurai -> None
    # Draw given Samurai details as text on local img variable
    def draw_sam(sm):
        # cv.putText(image, text, top-left, font, size, color, thickness)
        #   Font examples: cv.FONT_HERSHEY_SIMPLEX
        nonlocal TXT_LOC
        sz = 0.5
        clr = (0, 255, 0)
        thc = 2
        if sm is None:
            cv.putText(img, "Samurai is Dead", TXT_LOC, FONT, sz, clr, thc)
            return
        cv.putText(img, "Samurai is at ({0}, {1})"
                    .format(sm.position.x, sm.position.y),
                   TXT_LOC, FONT, sz, clr, thc)
        TXT_LOC = (TXT_LOC[0], TXT_LOC[1] + 25)
        cv.putText(img, "Samurai is heading {0}"
                .format("left" if sm.direction == Direction.L else 
                        ("right" if sm.direction == Direction.R else "Unknown")),
                   TXT_LOC, FONT, sz, clr, thc)
        TXT_LOC = (TXT_LOC[0], TXT_LOC[1] + 25)
        cv.putText(img, "Samurai is {0}"
                .format(("Standing" if sm.state == State.STAND else 
            ("In Air" if sm.state == State.JUMP else "Dead"))),
                   TXT_LOC, FONT, sz, clr, thc)
        TXT_LOC = (TXT_LOC[0], TXT_LOC[1] + 25)
        if acn is None:
            text = "Samurai should do nothing"
        else:
            hdrc = "left" if acn.direction == Direction.L else "right" 

            if type(acn) is Attack:
                text = "Samurai should Attack to the {0}".format("lower direction"
                        if acn.direction is None else hdrc)
            elif type(acn) is Move:
                text = "Samurai should Move {0} pixels to the {1} direction".format(
                        acn.goal, hdrc)
            elif type(acn) is Jump:
                text = "Samurai should Jump {0} pixels to the {1} direction".format(
                        acn.goal, hdrc + "-" + ("upper" if acn.vert_direction == 
                        VDirection.U else "lower"))
                
        cv.putText(img, text, TXT_LOC, FONT, sz, clr, thc)

    # Enemy -> None
    # Draw given Enemy details as text on local img variable
    def draw_enm(enm):
        sz = 0.4
        clr = (0, 255, 255)
        thc = 1
        dst = 15
        loc = (enm.position.x - (enm.size.w // 2), 
               enm.position.y - (dst * 5) - (enm.size.h // 4)) 
        cv.putText(img, "position  = ({0}, {1})".format(enm.position.x,
                                                        enm.position.y),
                   loc, FONT, sz, clr, thc)
        loc = (loc[0], loc[1] + dst)
        cv.putText(img, "direction = {0}"
                   .format("left" if enm.direction == Direction.L else 
                       ("right" if enm.direction == Direction.R else "Unknown")), 
                   loc, FONT, sz, clr, thc)
        loc = (loc[0], loc[1] + dst)
        cv.putText(img, "speed     = {0}".format(enm.speed),
                   loc, FONT, sz, clr, thc)
        loc = (loc[0], loc[1] + dst)
        cv.putText(img, "status    = {0}"
                .format(("Standing" if enm.state == State.STAND else 
                        ("In Air" if enm.state == State.JUMP else 
                        ("Dead" if enm.state == State.DEAD else "Flying")))),
                   loc, FONT, sz, clr, thc)

    # tuple of Platform -> None
    # Draw white line over each platform in mp on local img variable
    def draw_map(mp):
        mp = mp.get_pms_astpl()
        for pm in mp:
            cv.line(img, (pm.location.x, pm.location.y),
                         (pm.location.x + pm.length, pm.location.y),
                    (255, 255, 255), 5)


    if not (gs.samurai is None): 
        draw_rect(gs.samurai)

    draw_sam(gs.samurai)
    if not (gs.gb is None): draw_rect(gs.gb)
    if not (gs.enemies is None):
        for enm in gs.enemies:
            draw_rect(enm)
            draw_enm(enm)

    draw_map(gs.map)
    
    if not (pth == ()):
        prvp = None 
        for p in pth:
            draw_rect(p)
            if not (prvp is None):
                cv.line(img, (prvp.x, prvp.y), (p.x, p.y),
                        (0, 255, 0), 3)

            prvp = Point(p.x, p.y) 


    show_img(img)


         

# BGR Image, GameState -> GameState
# Detect the current GameState from given two consecutive
#   game BGR Images
#
# <Tested Implicitly>
#
def detect(img, gs):
    hsv  = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    sm   = fetch_sm(hsv, img, gs)
    enms = fetch_enms(hsv, img, gs)
    gb   = fetch_gb(hsv, gs)
    # Replace the oldest enemy with the new one
    onms = gs.old_enms[1:] + (gs.enemies,)
    # New GameState object to avoid mutation 
    ngs = GameState(MAP_1)
    ngs.samurai  = sm
    ngs.old_enms = onms 
    ngs.enemies  = enms
    ngs.gb       = gb

    return ngs

# HSV Image, Obj class -> tuple of Point
# Find objects that belong to given obj class
#   that appear in given image and return each
#   ones position relative to img
#
#   <Use "python tests.py fobjs_(sam or enms) <img number>" to test>
#
def find_objs(hsv, obj : Obj):
    # Find pixels that lie between given lower-upper range
    #   and return threshold image(mask) where pixels of
    #   specified color are white(255) and everything
    #   else is             black(  0)
    #
    mask = cv.inRange(hsv, obj.CR[0], obj.CR[1])
    mask = cv.dilate(mask, None, iterations=2)
    # Find bounding rectangles of each white shape in mask
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_SIMPLE)[0]
    # List of cv.Rectangle -> tuple of Point 
    # Convert each rectangle in given list to
    #  Point represeting the center of the rectangle 
    def helper(cs, acc):
        if len(cs) == 0:
            return acc
        else:
            first = cs[0]
            x, y, w, h = cv.boundingRect(first)
            x, y = (x + (w // 2)), (y + (h // 2))

            return helper(cs[1:], acc + (Point(x, y),))

    
    return helper(cnts, ())
    
    

# HSV Image, BGR Image, GameState -> Samurai, None
# Find and collect Samurai info from given BGR Images
#   return None if no enemy was found
#
# <Use "python tests.py fetch_sm <img #>" to test>
#
def fetch_sm(hsv, bgr, gs) -> Samurai:
    smp = find_objs(hsv, Samurai)
    # Samurai not found
    if smp == (): return None
    smp = smp[0]
    sm = Samurai()
    sm.position = smp
    sm.update_status(bgr, gs)
    
    return sm 

# HSV Image, BGR Image, GameState -> tuple of Enemy, None
# Find and collect all Enemy(s) info from given BGR Images
#   return None if no enemy was found
#
# <Use "python tests.py fetch_enms <img #>" to test>
#
def fetch_enms(hsv, bgr, gs):
    brdps = find_objs(hsv, Bird)
    brgps = find_objs(hsv, Burger)
    # No enemies found
    if brdps == ():
        if brgps == ():
            return None
        else:
            enms = tuple(map(lambda p: Burger(None, None, Burger.DFSP, 
                                              (p.x, p.y)), brgps))
    else:
        enms = tuple(map(lambda p: Bird(  None, None, Bird.DFSP, 
                                        (p.x, p.y)), brdps))
        if brgps == ():
            pass
        else:
            enms += tuple(map(lambda p: Burger(None, None, Burger.DFSP,
                                               (p.x, p.y)), brgps))




    
    # tuple of Enemy -> tuple of Enemy
    #  update status of each Enemy in given tuple
    def aux(toe):
        if len(toe) == 0:
            return 
        else:
            toe[0].update_status(None, gs)
            return aux(toe[1:])

    aux(enms)
     
    return enms
    


# HSV Image, GameState -> GoldBag, None
# Find and collect GoldBag info from given BGR Images
#
# <Use "python tests.py fetch_gb <img #>" to test>
#
def fetch_gb(hsv, gs):
    gbp = find_objs(hsv, GoldBag)
    # No gold bag found
    if gbp == (): return None
    gbp = gbp[0]
    gb = GoldBag()
    gb.position = gbp
    if gs.gb is None:
        gb.count = 15
        return gb
    else:
        gb.count = gs.gb.count
    
    if gs.gb.position.x == 0 and gs.gb.position.y == 0:
        pass
    # decrease gold bag count if it's position shifted
    #   more than 10 pixels else keep the old count
    elif math.dist((gs.gb.position.x, gs.gb.position.y),
                 (gb.position.x, gb.position.y)) > 10:
        gb.count = gs.gb.count - 1

    return gb



if __name__ == "__main__":
    main()
