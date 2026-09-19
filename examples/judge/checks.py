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
          fps=10.0, window_s=1.6, style=BROADCAST,
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

BY_KEY = {c.key: c for c in CHECKS}
SCENES = sorted({c.scene for c in CHECKS}, key=lambda s: [c.scene for c in CHECKS].index(s))
