/*
	Module: myconst.h	Date:	21.2.1990
	This file contains all the constants as defined by the

			Space Fortress Game

	Use: It is included in the source files as
	#include "myconst.h"
	and must be present in the current directory!
*/

#define UP 72
#define DOWN 80
#define LEFT 75
#define RIGHT 77
#define ESC -1
#define F1 0xF1     /* 241 decimal */
#define F2 0xF2     /* 242 decimal */
#define F3 0xF3     /* 243 decimal */
#define ENTER 13
#define REGULAR_CRAP 1
#define EXTENDED_CRAP 2
#define NO_INPUT 0
#define ON 1
#define OFF 0
#define NIL 0
#define RELEASED 2
#define THRUST 72
#define ROTATE_LEFT 75
#define ROTATE_RIGHT 77
#define FIRE 80
#define JSTK_TRESHOLD 4
#define ARC_CONV 0.0174527       /* degrees to radians */
#define MINE_COURSE_INTERVAL 10
#define FORT_LOCK_INTERVAL 20
#define SHELL_SPEED 7
#define MAX_NO_OF_MISSILES 6
#define DELAY 50

#define SHIP_COLOR YELLOW
#define HEX_COLOR GREEN
#define FRAME_COLOR GREEN
#define MINE_COLOR LIGHTCYAN
#define FRIEND_MINE_COLOR GREEN
#define FOE_MINE_COLOR LIGHTRED
#define FORT_COLOR LIGHTCYAN
#define MISSILE_COLOR BROWN
#define SHELL_COLOR LIGHTRED
#define TEXT_COLOR YELLOW
#define BACKGROUND_COLOR BLACK
#define TEXT_BACKGROUND BLUE
#define TEXT_LINE_COLOR MAGENTA

#define COLLISION_DIST 22
#define RESOURCE_DISPLAY_INTERVAL 120
#define NO_RESOURCE_DISPLAY_INTERVAL 40
#define MINE_SHIP_DISTANCE 200 /* in screen pixels */
#define MISSILE_FORT_TIME_LIMIT 40 /* after new mine appears */
#define MISSILE_FIRING_INTERVAL 5 /* loops between two consecutive missiles */
#define SHIP_EXPL_RADIUS 50 /* in screen pixels */
#define SHIP_GOOD_VELOCITY 2 /* for velocity points scoring */

#define BIG_HEXAGONE_SIZE_FACTOR  0.416  /*  10/24=0.416 of MaxX */
#define SMALL_HEXAGONE_SIZE_FACTOR 0.083 /*  2/24=0.0833 of MAxX */
#define SHIP_SIZE_FACTOR 0.041           /*  1/24=0.041 of MaxX  */
#define FORT_SIZE_FACTOR 0.05            /*  1.2/24=0.05 of MaxX */
#define MINE_SIZE_FACTOR 0.023           /*  0.55/24=0.023 of MaxX */
#define MISSILE_SIZE_FACTOR 0.03         /*  of MaxX */
#define SHELL_SIZE_FACTOR 0.03		 /*  of MaxX */
#define MAX_BONUS_WINDOWS 6     /* maximal number of bonus intervals */
#define MAX_DIR_PATH 30		/* maximal length of a DOS path */
#define MAXCHAR 15		/* maximal length of trainee name field */
#define MAXINT 3		/* maximal number if digits in ID#	*/
#define MAXFILES 20		/* maximal number of files in displayed */
	/*  trainee data directory (session or aiming directories) */

