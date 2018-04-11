#ifndef MYEXT_H
#define MYEXT_H
#include "myconst.h"
#include "myenums.h"
#include <cairo.h>
#include <time.h>

// Joystick stuff
//struct jstk_pos{
//  int x_center;
//  int x_left;
//  int x_right;
//  int y_center;
//  int y_up;
//  int y_down;
//  int no_push_button;
//  int top_push_button;
//  int lower_push_button;
//};

//struct aim_sess_results{
//	int mines;
//	int speed;
//	int score;
//};

// Addded for backwards compability
extern int Missile_X;
extern int Missile_Y;

extern char *Mine_Char;

#ifdef GUI_INTERFACE
extern cairo_t *SF_rgb_context;
extern cairo_surface_t *rgb_surface;
#endif
extern cairo_t *SF_canvas;
extern int Terminal_State_Flag;
extern cairo_surface_t *surface;
//extern cairo_font_options_t *font_options;

extern char Initialized_Graphics;

//extern cairo_path_t *PrevShip;
//extern cairo_path_t *PrevMissile[MAX_NO_OF_MISSILES];
//extern cairo_path_t *PrevFort;
//extern cairo_path_t *PrevMine;
//extern cairo_path_t *PrevShell;

extern int Terminal_State;

//extern int Ship_Should_Update;
extern int Bonus_Char_Should_Update;
//extern int Mine_Should_Update;
//extern int Fort_Should_Update;
//extern int Missile_Should_Update[MAX_NO_OF_MISSILES]; // Make this an array for all the missiles
//extern int Shell_Should_Update;
//extern int Mine_Type_Should_Update;
//// Panel text
//extern int Points_Should_Update;
//extern int Velocity_Should_Update;
//extern int Speed_Should_Update;
//extern int Vulner_Should_Update;
//extern int Interval_Should_Update;
//extern int Shots_Should_Update;
//extern int Control_Should_Update;
//
//extern int Mine_Type_Should_Clean;
//extern int Ship_Should_Clean;
//extern int Bonus_Char_Should_Clean;
//extern int Mine_Should_Clean;
//extern int Shell_Should_Clean;
//extern int Fort_Should_Clean;
//extern int Missile_Should_Clean[MAX_NO_OF_MISSILES];
//extern int Points_Should_Clean;
//extern int Velocity_Should_Clean;
//extern int Speed_Should_Clean;
//extern int Vulner_Should_Clean;
//extern int Interval_Should_Clean;
//extern int Shots_Should_Clean;
//extern int Control_Should_Clean;

extern const char *Char_Set[];
extern char Tmp_Char_Set[10][1];

extern const char *Foe_Menu[3];
extern const char *Friend_Menu[3];
extern char *Mine_Indicator;

extern int Explosion_Flag;
extern int Explosion_Step;
extern int ExpRadius;
extern int ExpX;
extern int ExpY;

extern int Jitter_Flag;
extern int Jitter_Step;

/* Not that good maybe because for example multiple missile can exist */
extern int Missile_X;
extern int Missile_Y;
extern int Missile_Heading; //

// Added variables that only were present in myvars.c,
extern int Freeze_Flag;
extern int Interval_Lower_Limit;
extern int Interval_Upper_Limit;
extern int Bonus_Indication_Index;
extern int No_Resource_Display_Interval;
extern int One_Game_Loops;
extern int Resource_Display_Interval;
extern char Restart_Flag;

// Used to be double longs
//extern clock_t t0;  /* time when FOE mine is born */
//extern clock_t t1;  /* double press interval start */
//extern clock_t t2;  /* double press interval end */
extern int t0;
extern int intv_t1;
extern int intv_t2;

		/* GAME PRAMATERS */
extern int Ship_Rotate_Step;
extern int Ship_Used;
extern int Ship_Max_Speed;
extern int Ship_Accel;
extern int Ship_Angular_Step;
extern int Missile_Speed;
extern int Mine_Wait_Loops;
extern int Mine_Live_Loops;
extern int Mine_Speed;
extern int Missile_Limit_Flag;

		/* VARIABLES */
extern int Ship_X_Pos;
extern int Ship_Y_Pos;
extern int Ship_X_Old_Pos;
extern int Ship_Y_Old_Pos;
extern float Ship_X_Speed;
extern float Ship_Y_Speed;
extern int Ship_Headings;
extern int Ship_Old_Headings;
extern int Rotate_Input;
extern int Accel_Input;
extern int Ship_Display_Update;
extern char End_Flag;

extern int Missile_X_Pos[MAX_NO_OF_MISSILES];
extern int Missile_Y_Pos[MAX_NO_OF_MISSILES];
extern float Missile_X_Speed[MAX_NO_OF_MISSILES];
extern float Missile_Y_Speed[MAX_NO_OF_MISSILES];
extern int Missile_Headings[MAX_NO_OF_MISSILES];
extern status Missile_Flag[MAX_NO_OF_MISSILES];
extern int Missiles_Counter;
extern int New_Missile_Flag;
extern missile_type Missile_Type;
extern int Missile_Vs_Mine_Only;
extern int Missile_Stock;

extern int Sound_Flag;
extern int Loop_Counter;
extern status Mine_Flag;
extern int Mine_X_Pos;
extern int Mine_Y_Pos;
extern int Mine_Headings;
extern int Mine_X_Speed;
extern int Mine_Y_Speed;
extern mine_type Mine_Type;

extern int Fort_Headings;
extern int Mine_Course_Count;
extern status Shell_Flag;
extern int Shell_X_Pos;
extern int Shell_Y_Pos;
extern int Shell_Headings;
extern int Shell_X_Speed;
extern int Shell_Y_Speed;

extern char *buffer1;
extern char *buffer2;
/* ************************************************************************************** */
/* Related to timing and waiting, for example for waiting for next screen update or between
keystrokes. Really old and dos exclusive tho. */

//extern void (interrupt far *oldfunc)(); /* address of INT 9 in interrupt vector */
//extern void (interrupt far *oldtik)(); /* address of INT 8 in interrupt vector */

/* ************************************************************************************** */
extern unsigned long Time_Counter;

extern int Wrap_Around_Flag;
extern int Points;
extern int Speed;
extern int Vulner_Counter;
extern int Key; /* keyboard input value */
extern int Lastkey; /* last keyboard input value */
extern int New_Input_Flag;
extern int Timing_Flag;
extern int Double_Press_Interval;
extern int Display_Interval_Flag;
extern int Check_Mine_Flag;
extern int Missile_Stock;
extern int Resource_Flag;
extern const char *Bonus_Char_Vector[];
extern bonus_character Bonus_Display_Flag;
extern int Bonus_Wasted_Flag;
extern int sax,sbx,scx,sdx; /* storage to save _AX,_BX,_CX,_DX */
extern int ax,ay;   /*  joystick's PORT A x and y input */
extern int bt;      /* joystick's pushbutton input value */
extern int Stick_Input; /* joystick's, receives values of THRUST,ROTATE_RIGHT
			   ROTATE_LEFT, and NIL                 */
extern int Fire_Input; /* joystick's input receives NIL and FIRE */
//extern struct jstk_pos jstk_clb; /* joystick calibration values */
extern int Fire_Button_Released;
extern mouse_button_type Mouse_Button;
extern mouse_button_type Last_Mouse_Button;
extern int Loop_Interval;

extern int    GraphDriver;	/* The Graphics device driver		*/
extern int    GraphMode;	/* The Graphics mode value		*/
extern double AspectRatio;	/* Aspect ratio of a pixel on the screen*/
extern int    MaxX, MaxY;	/* The maximum resolution of the screen */
extern int    MaxColors;	/* The maximum # of colors available	*/
extern int    ErrorCode;	/* Reports any graphics errors		*/
//extern struct palettetype palette;	/* Used to read palette info	*/

extern int Data_Line;		/* Data line location of control panel */
extern int Points_X;		/* Points value location within data line */
extern int Control_X;		/* Control value location within data line */
extern int Velocity_X;         /* Velocity value location within data line */
extern int Vulner_X;           /* Vulner value location within data line */
extern int IFF_X;              /* IFF value location within data line */
extern int Interval_X;         /* Interval value location within data line */
extern int Speed_X;            /* Speed value location within data line */
extern int Shots_X;            /* Shots value location within data line */
extern int Mines_X;            /* Mines value location within data line	*/
extern int Score_X;            /* Score value location within data line	*/

extern int Game_Counter;
extern int No_Of_Games;
extern int One_Game_Duration;
extern float Score;
extern int Points;
extern int Velocity;
extern int Control;
extern int Speed;
extern int Mines;
extern int No_Of_Bonus_Intervals;
extern int No_Of_Points_Bonus_Taken;
extern int No_Of_Missiles_Bonus_Taken;
extern int Ship_Damaged_By_Fortress;
extern int Ship_Damaged_By_Mines;
extern int Fortress_Destroyed;
extern int Normal_Game_Termination;
extern int Effect_Flag;

//extern struct file_header{
//	   int Session_Number;
//	   int Number_Of_Planned_Games;
//	   int One_Game_Duration;
//	   int Trainee_ID;
//	   char Trainee_Last_Name[MAXCHAR];
//	   char Trainee_First_Name[MAXCHAR];
//	   int File_Date[3];
//	   }header;
//
//extern struct games{
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

extern char filename[];
//extern char select[];
extern char curdir[];
extern int  scr_addr;
extern char path[];
extern int go_back;
extern game_type Game_Type;
extern int Check_Stick_Button_Flag;
extern struct aim_sess_results Aiming_Game_Results;
extern int Last_Missile_Hit;  /* to measure interval between two consecutive
			    hits of the fortress */
extern int Ship_Killings_Counter;
extern int Panel_Y_End;
extern int Panel_Y_Start;
extern int MaxY_Panel;
extern int Xmargin;
extern float GraphSqrFact; /* to convert Y to X in non-VGA environment */

extern char *bc[10];       /* bonus character bit map */
extern int rn;             /* random number - index of last bonus character */
extern int Bonus_Granted;     	/* flag to clear bonus announcement */
extern int Resource_Off_Counter;       /* Bous resource counters */
extern int Resource_On_Counter;
extern int No_Of_Bonus_Windows;
extern int Fort_Lock_Counter;


		/* FUNCTIONS */

extern float Fcos(int Headings_Degs); /* compute cos of 0 - 359 degrees */
extern float Fsin(int Headings_Degs); /* compute sin of 0 - 359 degrees */
extern void Draw_Ship (cairo_t *cr, int x, int y, int Headings, int size);
extern void Draw_Hexagone(cairo_t *cr, int X_Center, int Y_Center,int Hex_Size);
extern void Draw_Frame(cairo_t *cr);
extern void Draw_Fort(cairo_t *cr, int x, int y, int Headings, int size);
extern void Draw_Missile (cairo_t *cr, int x, int y, int Headings, int size, int missile_idx);
extern void Draw_Shell(cairo_t *cr, int x, int y, int Headings, int size);
extern void Draw_Mine (cairo_t *cr, int x, int y, int size);  /* x,y is on screen center location
					size is half diagonal           */

extern float Find_Headings(int x1, int y1, int x2, int y2);
extern void Clear_Interval();
extern void Update_Shots();
extern void Update_Speed();
/* extern Compute_Interval(struct time *Start, struct time *end); */
#endif
