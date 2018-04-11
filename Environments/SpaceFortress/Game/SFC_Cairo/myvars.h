#ifndef MYVARS_H
#define MYVARS_H
#include <cairo.h>
#include "myconst.h"
#include "myext.h"
#include "myenums.h"

cairo_t *SF_canvas;
cairo_surface_t *surface;
cairo_font_options_t *font_options;

cairo_path_t *PrevShip;
cairo_path_t *PrevMissile;
cairo_path_t *PrevFort;
cairo_path_t *PrevMine;

//int Ship_Should_Update;
//int Mine_Should_Update;
//int Fort_Should_Update;
//int Missile_Should_Update;

int Square_X = 0;
int Square_Y = 0;
int Square_Step;
int Square_Flag = KILL;

char Initialized_Graphics;

/* Not that good maybe because for example multiple missile can exist */
int Missile_X;
int Missile_Y;
int Missile_Heading;

struct jstk_pos{
  int x_center;
  int x_left;
  int x_right;
  int y_center;
  int y_up;
  int y_down;
  int no_push_button;
  int top_push_button;
  int lower_push_button;
};

struct aim_sess_results{
	int mines;
	int speed;
	int score;
};

//typedef enum { ALIVE, DEAD, KILL } status;
//typedef enum { FRIEND, FOE } mine_type;
//typedef enum { VS_FRIEND, VS_FOE, WASTED } missile_type;
//typedef enum { NOT_PRESENT, NON_BONUS, FIRST_BONUS, SECOND_BONUS } bonus_character;
//typedef enum { LEFT_BUTTON,CENTER_BUTTON,RIGHT_BUTTON,NONE } mouse_button_type;
//typedef enum { SPACE_FORTRESS, AIMING_TEST } game_type;

unsigned long t0;  /* time when FOE mine is born */
unsigned long t1;  /* double press interval start */
unsigned long t2;  /* double press interval end */



		/* PRAMATERS */
int One_Game_Duration=3; 	/****** in minutes 1-6 allowed */
int No_Of_Games=1;  /******** varies 1-9 */
int Resource_Display_Interval=RESOURCE_DISPLAY_INTERVAL;
int No_Resource_Display_Interval=NO_RESOURCE_DISPLAY_INTERVAL;
int Interval_Upper_Limit=SF_DELAY*20;
int Interval_Lower_Limit=SF_DELAY*5;
int Ship_Angular_Step=10;   /* Display increment in degrees */
int Ship_Max_Speed=5;        /* dots per loop */
int Ship_Accel=1;            /* dots/(loop*loop) */
int Mine_Speed=4;	   /* in screen dots per loop */
int Mine_Wait_Loops=80;
int Mine_Live_Loops=200;     /* new parameter */
int Missile_Speed=30;		/* new parameter */
int Missile_Limit_Flag=OFF; /******** missile borrowing is allowed */
int Shell_Speed=SHELL_SPEED; /* in dots/loop */
int Fort_Lock_Interval=FORT_LOCK_INTERVAL; /* in loops */
int Collision_Distance=COLLISION_DIST; /*********** in screen dots */

		/* VARIABLES */




int Bonus_Indication_Index=0; /* first entry in the Vector   crap****/
int Ship_X_Pos;
int Ship_Y_Pos;
int Ship_X_Old_Pos;
int Ship_Y_Old_Pos;
float Ship_X_Speed;
float Ship_Y_Speed;
int Ship_Headings;
int Ship_Old_Headings;
int Rotate_Input=0;
int Accel_Input=0;
int Ship_Display_Update=0;
char End_Flag=0;
char Restart_Flag=0;

int Missile_X_Pos[MAX_NO_OF_MISSILES];
int Missile_Y_Pos[MAX_NO_OF_MISSILES];
float Missile_X_Speed[MAX_NO_OF_MISSILES];
float Missile_Y_Speed[MAX_NO_OF_MISSILES];
int Missile_Headings[MAX_NO_OF_MISSILES];
status Missile_Flag[MAX_NO_OF_MISSILES];
int Missiles_Counter=0;
int New_Missile_Flag=OFF;
missile_type Missile_Type=VS_FRIEND; /* effective against friend mine */
int Missile_Vs_Mine_Only=OFF;

int Sound_Flag;
int Loop_Counter=0;
status Mine_Flag;
int Mine_X_Pos;
int Mine_Y_Pos;
int Mine_Headings;
int Mine_X_Speed;
int Mine_Y_Speed;
int Fort_Headings;
int Mine_Course_Count=MINE_COURSE_INTERVAL;
status Shell_Flag;
int Shell_X_Pos;
int Shell_Y_Pos;
int Shell_Headings;
int Shell_X_Speed;
int Shell_Y_Speed;
int Vulner_Counter=0;
char *buffer1;  /* blank eraser */
char *buffer2; /* bonus announcement */
//void (interrupt far *oldfunc)(); /* address of INT 9 in interrupt vector */
//void (interrupt far *oldtik)(); /* address of INT 8 in interrupt vector */
unsigned long Time_Counter=0;
int Key=0; /* keyboard input value */
int Lastkey=0; /* last keyboard input value */
int New_Input_Flag=OFF;
int Timing_Flag=OFF;
int Double_Press_Interval=0;
int Display_Interval_Flag=OFF;
int Check_Mine_Flag=OFF;
int Missile_Stock=100;
int Resource_Flag=OFF;
char Bonus_Char_Vector[10][1]={"$","[","?","%","&","!","#","@",">","/"};
bonus_character Bonus_Display_Flag=NOT_PRESENT;
int Bonus_Wasted_Flag=OFF;
int sax,sbx,scx,sdx; /* storage to save _AX,_BX,_CX,_DX */
int ax,ay;   /*  joystick's PORT A x and y input */
int bt;      /* joystick's pushbutton input value */
int Stick_Input=NIL; /* joystick's, receives values of THRUST,ROTATE_RIGHT
			   ROTATE_LEFT, and NIL                 */
int Fire_Input=NIL; /* joystick's input receives NIL and FIRE */
struct jstk_pos jstk_clb; /* joystick calibration values */
int Fire_Button_Released=ON; /* fire disabled until button released */
mouse_button_type Mouse_Button=NONE;
int Check_Stick_Button_Flag=OFF;

int    GraphDriver;		/* The Graphics device driver		*/
int    GraphMode;		/* The Graphics mode value		*/
double AspectRatio;		/* Aspect ratio of a pixel on the screen*/
int    MaxX, MaxY;		/* The maximum resolution of the screen */
int    MaxColors;		/* The maximum # of colors available	*/
int    ErrorCode;		/* Reports any graphics errors		*/
//struct palettetype palette;		/* Used to read palette info	*/

int Data_Line;		/* Data line location of control panel 		*/
int Points_X;		/* Points value location within data line 	*/
int Control_X;		/* Control value location within data line 	*/
int Velocity_X;         /* Velocity value location within data line 	*/
int Vulner_X;           /* Vulner value location within data line 	*/
int IFF_X;              /* IFF value location within data line  	*/
int Interval_X;         /* Interval value location within data line 	*/
int Speed_X;            /* Speed value location within data line 	*/
int Shots_X;            /* Shots value location within data line 	*/
int Mines_X;            /* Mines value location within data line	*/
int Score_X;            /* Score value location within data line	*/

int One_Game_Loops=3600; /*  number of loops in one game  3*60*20	*/
int Game_Counter=0;
float Score=0;
int Points=0;
int Velocity=0;
int Control=0;
int Speed=0;
int Mines=0;
int No_Of_Bonus_Intervals=6;
int No_Of_Points_Bonus_Taken=0;
int No_Of_Missiles_Bonus_Taken=0;
int Ship_Damaged_By_Fortress=0;
int Ship_Damaged_By_Mines=0;
int Fortress_Destroyed=0;
int Normal_Game_Termination=0;
int Freeze_Flag=0;

//struct file_header{
//	   int Session_Number;
//	   int Number_Of_Planned_Games;
//	   int One_Game_Duration;
//	   int Trainee_ID;
//	   char Trainee_Last_Name[MAXCHAR];
//	   char Trainee_First_Name[MAXCHAR];
//	   int File_Date[3];
//	   }header;
//
//struct games{
//	   int Score;
//	   int Points;
//	   int Velocity;
//	   int Control;
//	   int Speed;
//	   int No_Of_Bonus_Intervals;
//	   int No_Of_Points_Bonus_Taken;
//	   int No_Of_Missiles_Bonus_Taken;
//	   int No_Of_Ship_Damaged_By_Fortress;
//	   int No_Of_Times_Ship_Damaged_By_Mines;
//	   int No_Of_Times_Fortress_Distroyed_By_Trainee;
//	   int Normal_Game_Termination;
//	   }game;

char filename[MAX_DIR_PATH];
//char select[MAX_DIR_PATH];
char curdir[MAX_DIR_PATH];
int  scr_addr;
char path[MAX_DIR_PATH];
int go_back;


struct aim_sess_results Aiming_Game_Results;
game_type Game_Type=SPACE_FORTRESS;
int Last_Missile_Hit=0;  /* to measure interval between two consecutive
			    hits of the fortress */
int Ship_Killings_Counter=0;
int  Panel_Y_End;
int  Panel_Y_Start;
int MaxY_Panel;
int Xmargin;
float GraphSqrFact; /* to convert Y to X in non-VGA environment */

char *bc[10];       /* bonus character bit map */
int rn;             /* random number - index of last bonus character */
int Bonus_Granted=OFF;     	/* flag to clear bonus announcement */
int Resource_Off_Counter;       /* Bous resource counters */
int Resource_On_Counter;
int No_Of_Bonus_Windows=0;
int Fort_Lock_Counter=0;
int Effect_Flag=OFF;

#endif
