"""judge デモの「審判項目」定義と、テスト動画生成用のシーン記述。

各項目は 1 つの Noul (はい / いいえ / 判別できない) として VLM に問う。fps と窓長は項目ごとに変える
(安全・監視系は 2 fps × 4 秒、スポーツの反則は 10 fps × 1.6 秒)。
テスト動画は 5 秒の断片 (Wan2.2) を 3 本つないだ 15 秒: 問題あり = 通常 → 事象 → 通常 / 問題なし = 通常 × 3。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Check:
    key: str
    title: str            # 表示名
    scene: str            # シーンの区分 (UI のグループ)
    question: str         # VLM への英語の質問 (Question: ... Options: A. yes B. no C. cannot tell)
    fps: float = 2.0
    window_s: float = 4.0
    risk: str = "low"     # openvons.core.decide の危険度
    event_prompts: list[str] = field(default_factory=list)   # 事象あり (5 秒断片)
    normal_prompts: list[str] = field(default_factory=list)  # 事象なし
    style: str = ""       # 共通のカメラ・画質の書き方
    sport: str = ""              # スポーツの種目 (サブ場面)。「自動」ではシーンの種目を 1 問で判定し、合う種目の項目だけ見る
    positive: bool = False       # True = 「あった」が良い知らせの事象 (ヒット・ゴールなど)。表示を 検出/問題なし ではなく あり/なし にする
    long_window_s: float = 0.0   # >0 なら「前後の文脈」用の長い窓 (1 fps) も併用する。置く→離れる、取る→隠す→出る、外→中 など
    long_fps: float = 1.0


CCTV = "Fixed security camera footage, slightly high angle, realistic, natural lighting, no text, no watermark."
BROADCAST = "Broadcast sports camera, realistic, steady, no on-screen graphics."
SITE = "Construction site safety camera footage, realistic, daylight, wide shot, no text."

CHECKS: list[Check] = [
    Check("fight", "喧嘩・暴力", "店舗・街", "Are people fighting or physically attacking each other?", risk="high", style=CCTV,
          event_prompts=["Two men in a convenience store aisle shove each other and one throws a punch, the other stumbles back, aggressive body language.",
                         "On a night street two young men grab each other's collars and wrestle, one swings a fist, bystander steps back.",
                         "In a bar, a man pushes another man hard against the counter and they start punching, glasses fall."],
          normal_prompts=["Two men in a convenience store aisle chat and laugh, one pats the other's shoulder, then they browse the shelves.",
                          "On a night street two friends greet each other with a handshake and a hug, then walk together talking.",
                          "In a bar two men sit at the counter, toast with glasses and talk calmly."]),
    Check("shoplift", "万引き", "店舗・街", "Does a person put store merchandise into their own bag, pocket or clothing (not a basket or cart) and walk away without paying?", risk="medium", style=CCTV,
          event_prompts=["Close view in a small shop: a woman glances around, takes a cosmetics box from the shelf and clearly pushes it deep into her open handbag, closes the bag and walks toward the exit past the register.",
                         "Supermarket aisle, a man looks over his shoulder, opens his jacket and hides a bottle inside it against his chest, zips the jacket and walks away with his hands empty.",
                         "Convenience store, a teenager swings his backpack to the front, stuffs two snack packs into it while watching the clerk, zips it and walks out without paying."],
          normal_prompts=["In a small shop a woman picks up a cosmetics box, reads the label and puts it into her shopping basket.",
                          "A man in a supermarket aisle takes a bottle from the shelf and places it in his shopping cart, then pushes the cart on.",
                          "A teenager in a convenience store picks a snack pack and walks to the cash register and pays."]),
    Check("no_harness", "高所作業のハーネス未着用", "工事現場", "Is a worker at height (scaffold, roof, ladder) working without a safety harness attached?", risk="high", style=SITE,
          event_prompts=["A construction worker walks along a high scaffold platform wearing a helmet and work clothes but no safety harness, no lanyard, edge of the platform visible.",
                         "A man stands on a roof edge nailing panels, no harness or safety line, three stories up.",
                         "A worker climbs a tall ladder against a building and leans out to work on a wall, no harness, no fall protection."],
          normal_prompts=["A construction worker walks along a high scaffold platform wearing a helmet and a full-body safety harness with a lanyard clipped to the rail.",
                          "A man works on a roof edge wearing a safety harness connected by a rope to an anchor, helmet on.",
                          "A worker on a tall ladder wears a harness clipped to the structure and works on the wall carefully."]),
    Check("no_helmet", "ヘルメット未着用", "工事現場", "Is this a construction site where a worker's head is bare, with no hard hat (safety helmet) on?", risk="medium", style=SITE,
          event_prompts=["Construction site, two workers carry a steel pipe together; the worker on the left has NO helmet at all, his bare head and dark hair clearly visible, the other wears a yellow hard hat.",
                         "Close view of a bareheaded construction worker (no hard hat, hair uncovered) standing next to a small excavator on a dusty site, pointing at the ground.",
                         "A man in a high-visibility vest with his bare head fully visible (no helmet, no cap) walks under scaffolding carrying a bucket, other workers with helmets behind."],
          normal_prompts=["Construction site with two workers carrying steel pipes, both wearing yellow hard hats and vests.",
                          "A worker wearing a white hard hat operates a small excavator control panel on a construction site.",
                          "A man in a high-visibility vest and a hard hat walks under scaffolding carrying a bucket."]),
    Check("double_dribble", "バスケのダブルドリブル", "スポーツ", "Does the basketball player stop dribbling, hold the ball with both hands, and then start dribbling again (double dribble)?",
          fps=10.0, window_s=1.6, sport="バスケ", style=BROADCAST,
          event_prompts=["Indoor basketball court, a player dribbles the ball, stops and holds it firmly with both hands for a moment, then dribbles again and drives forward.",
                         "A basketball player in a gym dribbles, catches the ball in both hands, looks around, then resumes dribbling.",
                         "Street basketball, a player picks up his dribble with both hands, fakes a pass, then dribbles again toward the hoop."],
          normal_prompts=["Indoor basketball court, a player dribbles the ball continuously down the court and passes to a teammate.",
                          "A basketball player in a gym dribbles, stops, and shoots the ball toward the hoop in one motion.",
                          "Street basketball, a player dribbles past a defender and lays the ball up."]),
    Check("collapse", "人が倒れている", "介護・監視", "Is a person lying on the floor or collapsed?", risk="high", style=CCTV,
          event_prompts=["Office corridor, a middle-aged man walks, suddenly clutches his chest and collapses to the floor, lying motionless.",
                         "A hallway in a care facility, an elderly woman with a cane loses balance and falls, ending up lying on the floor.",
                         "A parking lot, a man slips and falls hard, then lies on the ground not moving."],
          normal_prompts=["Office corridor, a middle-aged man walks calmly holding documents and turns a corner.",
                          "A hallway in a care facility, an elderly woman walks slowly with a cane and sits down on a bench.",
                          "A parking lot, a man walks to his car, opens the door and gets in."]),
    Check("intrusion", "立入禁止区域への侵入", "工事現場", "Does someone climb over a fence, duck under a barrier tape, or squeeze through a gap to enter a restricted area (not using a gate normally)?", risk="medium", style=CCTV,
          event_prompts=["A person climbs over a chain-link fence with a KEEP OUT sign into a fenced construction yard at dusk.",
                         "A man ducks under a yellow barrier tape and walks into a closed area of a warehouse yard.",
                         "Someone squeezes through a gap in a gate into a restricted substation yard at night, looking around."],
          normal_prompts=["A person walks along the outside of a chain-link fence with a KEEP OUT sign and continues down the sidewalk at dusk.",
                          "A man stops in front of a yellow barrier tape, looks at it and walks away along the path.",
                          "A worker in a high-visibility vest walks along the inside of the fenced yard, checks a sign and walks on, gate closed."]),
    Check("fire", "火・煙", "介護・監視", "Is there fire or smoke that is not from normal cooking?", risk="high", style=CCTV,
          event_prompts=["A warehouse interior, thick gray smoke rises from a stack of boxes and flames appear at the bottom.",
                         "An office desk with a laptop, smoke starts pouring from the power strip under the desk and small flames flicker.",
                         "A kitchen, a pan on the stove bursts into large flames and black smoke fills the room, a person backs away."],
          normal_prompts=["A warehouse interior, a worker drives a forklift past stacks of boxes, clear air.",
                          "An office desk with a laptop, a person types and drinks coffee, everything normal.",
                          "A kitchen, a person stirs a pot on the stove with a little steam rising, calm cooking."]),
    Check("phone_driving", "運転中のスマホ操作", "道路", "Is the driver looking at or holding a smartphone while the vehicle is moving?", risk="medium", style="In-cabin driver monitoring camera, realistic, daytime, no text.",
          event_prompts=["Driver's seat view, a man drives on a city road while holding a smartphone in his right hand and looking down at the screen, typing.",
                         "A woman driving a car glances repeatedly at a phone in her hand near the steering wheel, road moving outside.",
                         "A delivery driver scrolls on a smartphone held above the steering wheel while the van is moving."],
          normal_prompts=["Driver's seat view, a man drives on a city road with both hands on the steering wheel, looking ahead.",
                          "A woman driving a car keeps her eyes on the road and adjusts the mirror, phone not visible.",
                          "A delivery driver drives a van with both hands on the wheel, checking the side mirror."]),
    Check("abandoned_bag", "放置された荷物", "店舗・街", "Is there a bag or suitcase left unattended on the floor with its owner walking away from it or gone?", risk="medium", style=CCTV,
          event_prompts=["Train station concourse, a man sets a black backpack down next to a pillar, looks around and walks away without it.",
                         "Airport waiting area, a woman leaves a suitcase beside a bench and walks off toward the exit, the suitcase stays.",
                         "Shopping mall bench, a person puts a shopping bag under the bench and leaves quickly."],
          normal_prompts=["Train station concourse, a man with a black backpack on his shoulders checks the departure board and walks on.",
                          "Airport waiting area, a woman sits beside her suitcase, then pulls it along with her when she leaves.",
                          "Shopping mall bench, a person sits with a shopping bag, then picks it up and walks away with it."]),
]

VARIANTS = [
    "",
    " Camera slightly lower and closer than a typical security camera. Late afternoon light.",
    " Wide shot from a corner-mounted camera, fluorescent lighting, a few other people in the background.",
    " Overcast daylight, handheld-looking but steady camera, muted colors.",
]

# 学習 head の入力 (質問方式の 30 logit) に使う 10 項目。features.npz と head_*.pt はこの順に依存するので、項目を足すときはここには足さない
HEAD_KEYS = ["fight", "shoplift", "no_harness", "no_helmet", "double_dribble", "collapse", "intrusion", "fire", "phone_driving", "abandoned_bag"]

CHECKS.append(
    Check("near_miss", "ヒヤリハット (飛び出し・急接近)", "道路",
          "Near miss: does a vehicle, motorbike, cyclist or pedestrian suddenly move into the path of the camera (the car or person filming) or very close to it, so that a collision is about to happen unless someone brakes, stops or swerves?",
          fps=2.0, window_s=3.0, long_window_s=8.0, risk="high", style="Dashcam footage from a car driving on a Japanese road, realistic, no text.",
          event_prompts=["A cyclist darts out from a side street directly in front of the car, the car brakes hard.",
                         "A child runs out from between parked cars into the road just ahead of the car.",
                         "A car pulls out of a parking lot without stopping and cuts across the lane right in front."],
          normal_prompts=["The car drives along a residential street, a cyclist rides on the left edge keeping its line.",
                          "The car waits at a crossing while pedestrians walk across, then proceeds slowly.",
                          "The car follows another car through a curve at a safe distance."]))

# ---- スポーツ (10 fps × 2.4 秒 = 24 フレーム。動作の判定は 10 fps 必要) ----
SPORTS = "Broadcast-style sports footage from a single fixed camera, realistic, natural stadium or field lighting, no text, no scoreboard graphics."
CHECKS += [
    Check("baseball_hit", "野球のヒット", "スポーツ",
          "Does the batter hit the pitched ball into the field of play (a fair hit, not a swing and miss, a foul, or a taken pitch)?",
          fps=10.0, window_s=2.4, sport="野球", risk="low", positive=True, style=SPORTS,
          event_prompts=["Baseball game, camera behind home plate: the pitcher throws, the batter swings and hits a sharp line drive into the outfield, drops the bat and runs to first base.",
                         "Baseball game, side view of home plate: the batter connects with the pitch and the ball flies over the infield, the batter sprints toward first base.",
                         "Baseball game, high camera behind the backstop: the batter hits a ground ball through the infield into the outfield grass and runs."],
          normal_prompts=["Baseball game, camera behind home plate: the pitcher throws, the batter swings and misses, the catcher catches the ball.",
                          "Baseball game, side view of home plate: the batter watches the pitch go by without swinging, the catcher throws it back to the pitcher.",
                          "Baseball game, high camera behind the backstop: the batter steps out of the box, adjusts the helmet and taps the bat on the plate, the pitcher waits."]),
    Check("golf_swing_fault", "ゴルフのスイングの欠点", "スポーツ",
          "Does the golf swing show an obvious fault: losing balance or stumbling after the swing, the head moving a lot, the body standing up early during the downswing, or a wild over-swing?",
          fps=10.0, window_s=2.4, sport="ゴルフ", risk="low", style=SPORTS.replace("stadium or field", "golf course"),
          event_prompts=["Golf driving range, side view: a golfer takes a wild over-swing, loses balance and stumbles a step sideways after hitting the ball.",
                         "Golf course tee box, view from behind: a golfer lifts the head and straightens the body early during the downswing, topping the ball, and staggers.",
                         "Golf practice, side view: a golfer sways heavily, the head moves far off the ball during the backswing, and the follow-through is off balance."],
          normal_prompts=["Golf driving range, side view: a golfer makes a smooth, balanced swing and holds a steady finish position watching the ball.",
                          "Golf course tee box, view from behind: a golfer swings with a stable head and posture and finishes balanced on the front foot.",
                          "Golf practice, side view: a golfer takes a relaxed practice swing, then addresses the ball calmly."]),
    Check("soccer_goal", "サッカーのゴール", "スポーツ",
          "Is a goal scored (the ball goes past the goalkeeper and into the goal net)?",
          fps=10.0, window_s=2.4, sport="サッカー", risk="low", positive=True, style=SPORTS,
          event_prompts=["Soccer match, camera behind the goal: a striker shoots from the edge of the box, the keeper dives and the ball flies into the net.",
                         "Soccer match, side view of the penalty area: a header from a corner kick goes into the goal past the keeper, the net ripples.",
                         "Soccer match, high wide camera: a low shot rolls into the corner of the goal, the goalkeeper on the ground, players raise their arms."],
          normal_prompts=["Soccer match, camera behind the goal: a striker shoots and the goalkeeper catches the ball cleanly.",
                          "Soccer match, side view of the penalty area: players pass the ball around outside the box, a defender clears it.",
                          "Soccer match, high wide camera: a shot hits the post and bounces away, players chase the rebound."]),
    Check("basketball_shot_made", "バスケのシュート成功", "スポーツ",
          "Does the shot go in (the ball passes down through the hoop and net)?",
          fps=10.0, window_s=2.4, sport="バスケ", risk="low", positive=True, style=SPORTS,
          event_prompts=["Basketball game, camera behind the baseline: a player takes a jump shot and the ball swishes through the net.",
                         "Basketball game, side view of the court: a player drives to the basket and lays the ball in, it drops through the hoop.",
                         "Basketball game, high camera: a three-point shot arcs and goes cleanly through the net."],
          normal_prompts=["Basketball game, camera behind the baseline: a player takes a jump shot, the ball hits the rim and bounces out.",
                          "Basketball game, side view of the court: players pass the ball around the perimeter, the defense shifts.",
                          "Basketball game, high camera: a player dribbles up the court and calls a play, no shot is taken."]),
    Check("soccer_handball", "サッカーのハンド", "スポーツ",
          "Does a player other than the goalkeeper stop or touch the ball with a hand or arm?",
          fps=10.0, window_s=2.4, sport="サッカー", risk="low", style=SPORTS,
          event_prompts=["Soccer match, side view: a defender in a dark kit reaches out and stops the ball with the hand, players around react and appeal.",
                         "Soccer match, camera behind the goal: a defender blocks a shot with an outstretched arm, the ball hits the arm clearly.",
                         "Soccer match, midfield view: a player catches the ball with both hands to stop a pass, then drops it."],
          normal_prompts=["Soccer match, side view: a defender blocks the ball with the chest and knee, then passes it away.",
                          "Soccer match, camera behind the goal: the goalkeeper catches the ball with both hands, defenders run out.",
                          "Soccer match, midfield view: players pass and control the ball with their feet."]),
    Check("running_form_fault", "ランニングフォームの問題", "スポーツ",
          "Does the runner show an obvious form problem: the foot landing far ahead of the body with a heavy heel strike, a strongly hunched or leaning-back upper body, or arms swinging across the body?",
          fps=10.0, window_s=2.4, sport="ランニング", risk="low", style="Side-view running footage from a fixed camera at a track or park path, realistic, no text.",
          event_prompts=["A runner on a track over-strides, landing heel first far in front of the body, upper body hunched forward, arms swinging across the chest.",
                         "A runner on a park path leans far back with the chest up, feet slapping down ahead of the body, arms flailing.",
                         "A runner on a track with the shoulders rolled forward and the head down, bouncing high with each heavy heel landing."],
          normal_prompts=["A runner on a track runs with an upright relaxed posture, landing under the body, arms swinging straight forward and back.",
                          "A runner on a park path jogs at a steady pace with a slight forward lean from the ankles and quiet footsteps.",
                          "A runner on a track stretches, then runs smoothly with relaxed shoulders."]),
]

# シーン種別 (分母) の判定: カットで区切った区間ごとに 1 回聞き、その区間ではシーンの合う項目だけを判定する
SCENE_OPTIONS = [
    ("店舗・街", "a shop, station concourse, street or other public place seen from a fixed security camera"),
    ("工事現場", "a construction site, factory or work at height"),
    ("スポーツ", "a sports game or practice"),
    ("介護・監視", "an indoor room, office, care facility or home"),
    ("道路", "a road, parking lot or traffic scene seen from a moving car (dashcam), from a walking person, or from a traffic camera, even with captions overlaid"),
    ("その他", "a title card or text-only screen with no live footage, or something else"),
]
SCENE_QUESTION = "What kind of footage is this?"

# サブ場面: スポーツと判定されたシーンでは種目を 1 問で決め、その種目の項目だけを見る (バスケの動画にサッカーのゴールを聞かない)
SUB_SCENE = {
    "スポーツ": ("Which sport is being played?", [
        ("バスケ", "basketball"), ("サッカー", "soccer (football)"), ("野球", "baseball or softball"), ("ゴルフ", "golf"),
        ("ランニング", "running, jogging or track athletics"), ("その他", "another sport or cannot tell"),
    ]),
}


def scene_of(c: "Check") -> str:
    """項目が意味を持つ場面のラベル (サブ場面があれば 場面/種目)。"""
    return f"{c.scene}/{c.sport}" if c.sport else c.scene

BY_KEY = {c.key: c for c in CHECKS}
SCENES = sorted({c.scene for c in CHECKS}, key=lambda s: [c.scene for c in CHECKS].index(s))
