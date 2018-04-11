from enum import Enum
import os

# Key bindings: http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/key-names.html
# Look at '.keysym_num'
class KeyMap(Enum):
	LEFT=65361
	UP=65362
	RIGHT=65363
	SHOOT=32

# Global variables to make syntax easier 
UP = KeyMap.UP.value
LEFT = KeyMap.LEFT.value
RIGHT = KeyMap.RIGHT.value
SHOOT = KeyMap.SHOOT.value

# Scripts of length 3
class ScriptsSF_3(Enum):
        # Sample scripts for now
        # Script 1: Move left 2 times then shoot
        SCRIPT1=[LEFT,LEFT,SHOOT]
        # Script 2: Move right 2 times then shoot
        SCRIPT2=[RIGHT,RIGHT,SHOOT]
        # Script 3: Move forward 3 times
        SCRIPT3=[UP,UP,UP]
        # Script 4: Move left then forward twice
        SCRIPT4=[LEFT,UP,UP]
        # Script 5: Move Right then forward twice
        SCRIPT5=[RIGHT,UP,UP]

# SCripts of length 9
class ScriptsSF_9(Enum):
	# Sample scripts for now
	# Script 1: Move left 2 times then shoot
	SCRIPT1=[LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,SHOOT,SHOOT,SHOOT]
	# Script 2: Move right 2 times then shoot
	SCRIPT2=[RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,SHOOT,SHOOT,SHOOT]
	# Script 3: Move forward 3 times
	SCRIPT3=[UP,UP,UP,UP,UP,UP,UP,UP,UP]
	# Script 4: Move left then forward twice
	SCRIPT4=[LEFT,LEFT,LEFT,UP,UP,UP,UP,UP,UP]
	# Script 5: Move Right then forward twice
	SCRIPT5=[RIGHT,RIGHT,RIGHT,UP,UP,UP,UP,UP,UP]

class ScriptsSFC_3(Enum):
        # Script 1: Move left 2 times then forward
        SCRIPT1=[LEFT,LEFT,UP]
        # Script 2: Move right 2 times then forward
        SCRIPT2=[RIGHT,RIGHT,UP]
        # Script 3: Move right 3 times
        SCRIPT3=[RIGHT,RIGHT,RIGHT]
        # Script 4: Move left 3 times
        SCRIPT4=[LEFT,LEFT,LEFT]
        # Script 5: Move forward then left twice
        SCRIPT5=[UP,LEFT,LEFT]
        # Script 6: Move forward then right twice
        SCRIPT6=[UP,RIGHT,RIGHT]
        # Script 7: Move forward thrice
        SCRIPT7=[UP,UP,UP]

class ScriptsSFC_9(Enum):
        # Script 1: Move left 2 times then forward
        SCRIPT1=[LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,UP,UP,UP]
        # Script 2: Move right 2 times then forward
        SCRIPT2=[RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,UP,UP,UP]
        # Script 3: Move right 3 times
        SCRIPT3=[RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT]
        # Script 4: Move left 3 times
        SCRIPT4=[LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT]
        # Script 5: Move forward then left twice
        SCRIPT5=[UP,UP,UP,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT]
        # Script 6: Move forward then right twice
        SCRIPT6=[UP,UP,UP,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT]
        # Script 7: Move forward thrice
        SCRIPT7=[UP,UP,UP,UP,UP,UP,UP,UP,UP]

class ScriptsAIM_3(Enum):
        # Script 1: Move left 2 times then shoot
        SCRIPT1=[LEFT,LEFT,SHOOT]
        # Script 2: Move right 2 times then shoot
        SCRIPT2=[RIGHT,RIGHT,SHOOT]
        # Script 3: Move right 3 times
        SCRIPT3=[RIGHT,RIGHT,RIGHT]
        # Script 4: Move left 3 times
        SCRIPT4=[LEFT,LEFT,LEFT]
        # Script 5: Move left then shoot twice
        SCRIPT5=[LEFT,SHOOT,SHOOT]
        # Script 6: Move right then shoot twice
        SCRIPT6=[RIGHT,SHOOT,SHOOT]
        # Script 7: Shoot thrice
        SCRIPT7=[SHOOT,SHOOT,SHOOT]

class ScriptsAIM_3_All(Enum):
# Shoot twice and then:
	# shoot
	SCRIPT1=[SHOOT,SHOOT,SHOOT]
	# move left
	SCRIPT2=[SHOOT,SHOOT,LEFT]
	# move right
	SCRIPT3=[SHOOT,SHOOT,RIGHT]
# Shoot once, then move left and then:
	# shoot
	SCRIPT4=[SHOOT,LEFT,SHOOT]
	# move left
	SCRIPT5=[SHOOT,LEFT,LEFT]
	# move rigth
	SCRIPT6=[SHOOT,LEFT,RIGHT]
# Shoot once, then move right and then:
	# shoot
	SCRIPT7=[SHOOT,RIGHT,SHOOT]
	# move right
	SCRIPT8=[SHOOT,RIGHT,RIGHT]
	# move left
	SCRIPT9=[SHOOT,RIGHT,LEFT]
# Move left twice and then:
	# move left
	SCRIPT10=[LEFT,LEFT,LEFT]
	# shoot
	SCRIPT11=[LEFT,LEFT,SHOOT]
	# move right
	SCRIPT12=[LEFT,LEFT,RIGHT]
# Move left, then shoot and then:
	# move left
	SCRIPT13=[LEFT,SHOOT,LEFT]
	# shoot
	SCRIPT14=[LEFT,SHOOT,SHOOT]
	# move right
	SCRIPT15=[LEFT,SHOOT,RIGHT]
# Move left, then right and then:
	# move left
	SCRIPT16=[LEFT,RIGHT,LEFT]
	# shoot
	SCRIPT17=[LEFT,RIGHT,SHOOT]
	# move right
	SCRIPT18=[LEFT,RIGHT,RIGHT]
# Move right twice and then:
	# move right
	SCRIPT19=[RIGHT,RIGHT,RIGHT]
	# shoot
	SCRIPT20=[RIGHT,RIGHT,SHOOT]
	# move left
	SCRIPT21=[RIGHT,RIGHT,LEFT]
# Move right, then shoot and then:
	# move right
	SCRIPT22=[RIGHT,SHOOT,RIGHT]
	# shoot
	SCRIPT23=[RIGHT,SHOOT,SHOOT]
	# move left
	SCRIPT24=[RIGHT,SHOOT,LEFT]
# Move right, then go left and then:
	# move right
	SCRIPT25=[RIGHT,LEFT,RIGHT]
	# shoot
	SCRIPT26=[RIGHT,LEFT,SHOOT]
	# move left
	SCRIPT27=[RIGHT,LEFT,LEFT]

class ScriptsAIM_9(Enum):
        # Script 1: Move left 2 times then shoot
        SCRIPT1=[LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,SHOOT,SHOOT,SHOOT]
        # Script 2: Move right 2 times then shoot
        SCRIPT2=[RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,SHOOT,SHOOT,SHOOT]
        # Script 3: Move right 3 times
        SCRIPT3=[RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT,RIGHT]
        # Script 4: Move left 3 times
        SCRIPT4=[LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT,LEFT]
        # Script 5: Move left then shoot twice
        SCRIPT5=[LEFT,LEFT,LEFT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT]
        # Script 6: Move right then shoot twice
        SCRIPT6=[RIGHT,RIGHT,RIGHT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT]
        # Script 7: Shoot thrice
        SCRIPT7=[SHOOT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT,SHOOT]

class Games(Enum):
	SFS="SFS"
	SF="SF"
	SFC="SFC"
	AIM="AIM"

class RenderMode(Enum):
	HUMAN="human"
	MINIMAL="minimal"
	RGB_ARRAY="rgb_array"

class RenderSpeed(Enum):
	# actually more of a render delay than speed 
	DEBUG=0
	SLOW=42
	MEDIUM=20
	FAST=8

class EnableScripts(Enum):
        ON = "on"
        OFF = "off"

class ScriptLength(Enum):
	THREE = 3
	NINE = 9

class AllCombinations(Enum):
	ON = "on"
	OFF = "off"

# GAME SETTINGS FOR RUN.PY
#GAME=Games.SFC
#RENDER_MODE=RenderMode.HUMAN
#RENDER_SPEED=RenderSpeed.DEBUG
#LIBRARY_NAME="_frame_lib"
#LIBRARY_PATH= '/home/victorgarcia/work/Environments/SpaceFortress/gym_space_fortress/envs/space_fortress/shared'#os.path.join()str(os.path.dirname(os.path.realpath(__file__))) + "/shared"
#GAME_VERSION='v0'
#
## OVERALL SETTINGS
#SCRIPTS = EnableScripts.OFF
#SCRIPT_LENGTH = ScriptLength.THREE # Should be three if scripts is off
#FRAMESKIP=SCRIPT_LENGTH.value
#ALL_COMBINATIONS = AllCombinations.OFF
#
#DEFAULT_RENDER_MODE=RenderMode.RGB_ARRAY.value
#DEFAULT_MAXSTEPS=2500000
#DEFAULT_TIMES=100
#RECORD=False
#STATS=False
#das=4
