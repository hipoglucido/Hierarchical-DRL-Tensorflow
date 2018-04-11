#ifndef MYENUMS_H
#define MYENUMS_H
#include "myconst.h"


//typedef enum { NOT_PRESENT, FIRST_TIME, SECOND_TIME } bonus_character;
// Added them all in a big enum
typedef enum { NOT_PRESENT, FIRST_TIME, SECOND_TIME, NON_BONUS, FIRST_BONUS, SECOND_BONUS } bonus_character;
typedef enum { ALIVE, DEAD, KILL } status;
typedef enum { FRIEND, FOE } mine_type;
typedef enum { VS_FRIEND, VS_FOE, WASTED } missile_type;
typedef enum { LEFT_BUTTON,CENTER_BUTTON,RIGHT_BUTTON,NONE } mouse_button_type;
typedef enum { SPACE_FORTRESS, AIMING_TEST } game_type;

extern struct file_header{
	   int Session_Number;
	   int Number_Of_Planned_Games;
	   int One_Game_Duration;
	   int Trainee_ID;
	   char Trainee_Last_Name[MAXCHAR];
	   char Trainee_First_Name[MAXCHAR];
	   int File_Date[3];
	   }header;

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
//
#endif
