/* ******************   Handle_Mine and all others elements   ***************** */
#include <stdio.h>
#include <stdlib.h>
//#include <graphics.h>
//#include <process.h>
//#include <bios.h>
//#include <alloc.h>
//#include <dos.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

//#include <cairo.h>

//#include "myconst.h"
//#include "myext.h"
//#include "myvars.h"

// Added
//#include "DE.h"
#include "HM.h"
#include "RS.h"

mine_type Mine_Type;
int Mine_Alive_Counter=0;
int Mine_Dead_Counter=0;
int Missile_Delay_Counter=0;


void Move_Ship()
{
  Ship_Old_Headings=Ship_Headings;
  Ship_X_Old_Pos=Ship_X_Pos;
  Ship_Y_Old_Pos=Ship_Y_Pos;
  Ship_Display_Update=0; /* do not refresh if no movement */

  if (Rotate_Input!=0)      /* if ship rotates */
    {
       Ship_Display_Update=1;  /* at least rotates */
       Ship_Headings=Ship_Headings + Rotate_Input*Ship_Angular_Step;
       if (Ship_Headings<0) Ship_Headings= 359+Ship_Headings-1;
       if (Ship_Headings>359) Ship_Headings= Ship_Headings-359-1;
       Rotate_Input=0;        /* reset input */
    }

}

void Fire_Shell()
{
  Shell_X_Pos=MaxX/2.0+0.5*SMALL_HEXAGONE_SIZE_FACTOR*MaxX*Fsin(Fort_Headings);
  Shell_Y_Pos=MaxY/2.0-0.5*SMALL_HEXAGONE_SIZE_FACTOR*Fcos(Fort_Headings);
  Shell_Headings=Fort_Headings;
  Shell_X_Speed=SHELL_SPEED*Fsin(Shell_Headings);
  Shell_Y_Speed=-SHELL_SPEED*Fcos(Shell_Headings);
//  Draw_Shell(cr, Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
//				SHELL_SIZE_FACTOR*MaxX);  /* first time */ // First time??
//	stroke_in_clip(cr);
//	Shell_Should_Update = 1; // first time apperantly
//	Shell_Should_Clean = 1;
//  sound(800);
//  Sound_Flag=6;
}


void Handle_Speed_Score()
{
//  struct timeval tDiff;
  int dts;

  dts=0;


	/* mine bonus for any type */
	if(Mine_Alive_Counter>=Mine_Live_Loops)
	{
		dts=-100;
	}
	else
	{

		if(Mine_Alive_Counter<=20)
		{
			 dts=80;
		}
		else if(Mine_Alive_Counter<=40)
		{
			dts=60;
		}
		else if(Mine_Alive_Counter<=60)
		{
			dts=40;
		}
		else if(Mine_Alive_Counter<=80)
		{
			dts=20;
		}
		else if(Mine_Alive_Counter<=100)
		{
			dts=0;
		}
		else if(Mine_Alive_Counter<=120)
		{
			dts=-10;
		}
		else if(Mine_Alive_Counter<=140)
		{
			dts=-40;
		}
		else if(Mine_Alive_Counter<=160)
		{
			dts=-60;
		}
		else if(Mine_Alive_Counter<=180)
		{
			dts=-80;
		}
		else // if(Mine_Alive_Counter<=200)
		{
			dts=-100;
		}
	}

  Speed=Speed+dts;
}




int randrange(int min, int max)
{
	return (rand() % (max + 1 - min)) + min;
}

void Generate_Mine()
{
  int a;
  do
  {
//    Mine_X_Pos=random(MaxX); // Maybe not available, what does it do?
//    Mine_Y_Pos=random(MaxY);
		Mine_X_Pos=randrange(0, MaxX);
		Mine_Y_Pos=randrange(0, MaxY);
    a=sqrt(pow(Mine_X_Pos-Ship_X_Pos,2)+pow(Mine_Y_Pos-Ship_Y_Pos,2));
  } while(a < 0.5*MaxX );  /* repeat until distance exceeds min. */

//  Draw_Mine(cr,Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);  /* draw mine first time */
//	Mine_Should_Clean = 1;
}


void Handle_Mine()
{
 switch(Mine_Flag)
 {
  case KILL  : {
		  Handle_Speed_Score();
//		  Draw_Mine(cr, Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
//			clear_prev_path(cr, PrevMine);
//			Mine_Should_Clean = 1;
							/* erase mine */
		  Mine_Flag=DEAD;
		  Mine_Dead_Counter=0;
		  Missile_Type=VS_FRIEND;
		  Missile_Vs_Mine_Only=OFF;
		  Timing_Flag=OFF;
//		  Clear_Mine_Type(cr); /* clear mine type display */
//			Mine_Type_Should_Clean = 1;
//		  Clear_Interval(); // Double press checking
		  break;
		}
  case DEAD   : {
		  if(Mine_Dead_Counter++ >= Mine_Wait_Loops)
		    {
		      Generate_Mine();
		      Mine_Flag=ALIVE;
		      Mine_Alive_Counter=0;
		    }
		   break;
		}
  case ALIVE  : {
		  if(Mine_Alive_Counter++ >= Mine_Live_Loops)
			{
		  	Mine_Flag=KILL;
			}
    }

 } /* end switch */
}



void Generate_Aim_Mine()
{
    float radius;
    float mine_distance;
    float mine_angle;

    radius=((float)MaxX)/2.2;
    mine_angle=((float)randrange(0, 15))*22.5;
    if(mine_angle>338.0) mine_angle=0.0;
    mine_distance=radius/2.0+ ((float)randrange(0, 1))*radius/2.0;

    Mine_X_Pos=((float)MaxX)/2.0 + mine_distance*Fsin(mine_angle);
    Mine_Y_Pos=((float)MaxY)/2.0 - mine_distance*Fcos(mine_angle);
//    else Mine_Y_Pos=MaxY/2 - mine_distance*Fcos(mine_angle)/GraphSqrFact;
		     /* Y/X square ratio */

//    Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
	     /* draw mine */
}

void Handle_Aim_Mine()
{

 switch(Mine_Flag)
 {

  case KILL  : {
//			printf("KILL");
			  Handle_Speed_Score();
//		  Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
							/* erase mine */
		  Mine_Flag=DEAD;
		  Mine_Dead_Counter=0;
		  break;
		}
  case DEAD   : {
//			printf("Dead -- Mine_dead_counter: %d Mine_Wait_Loops: %d", Mine_Dead_Counter, Mine_Wait_Loops);
		  if(Mine_Dead_Counter++ >= Mine_Wait_Loops)
		    {
		      Generate_Aim_Mine();
		      Mine_Flag=ALIVE;
		      Mine_Alive_Counter=0;
		    }
		   break;
		}
  case ALIVE  : {
		  if(Mine_Alive_Counter++ >= Mine_Live_Loops)
		  Mine_Flag=KILL;
		}
 } /* end switch */
}




void Fire_Missile(int Index)
{
	Missile_Headings[Index]=Ship_Headings;
	Missile_X_Pos[Index]=Ship_X_Pos;
	Missile_Y_Pos[Index]=Ship_Y_Pos;
	Missile_X_Speed[Index]= Missile_Speed*Fsin(Ship_Headings);
	Missile_Y_Speed[Index]=-Missile_Speed*Fcos(Ship_Headings);
//	Missile_Should_Update[Index] = 1;
// Draw_Missile(cr,Missile_X_Pos[Index],Missile_Y_Pos[Index],
//	      Missile_Headings[Index],MISSILE_SIZE_FACTOR*MaxX);
//	stroke_in_clip(cr);

							/* first time */
// sound(1000);
// Sound_Flag=4;
}

// Use PrevMine etc. in these kinds of functions (the moves/handles and alike)
void Handle_Missile()
{
 int i;
		/* update all existing missiles */
 for(i=0;i<MAX_NO_OF_MISSILES;i++)
    if(Missile_Flag[i] != DEAD)
      switch(Missile_Flag[i])
      {
	 case KILL  : {
//			clear_prev_path(cr,PrevMissile);
//			Missile_Should_Clean[i] = 1;
//			Draw_Missile(cr, Missile_X_Pos[i],Missile_Y_Pos[i],
//			     Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);  /* erase missile */
			Missile_Flag[i]=DEAD;
			Missiles_Counter--;
			break;
		      }

	 case ALIVE : {
			 if((Missile_X_Pos[i]<=41) || (Missile_X_Pos[i]>=MaxX-41)
			 || (Missile_Y_Pos[i]<=41) || (Missile_Y_Pos[i]>=MaxY-41))
				{
			    	Missile_Flag[i]=KILL;
					Score = -0.1;
				}
			 else
			  {
//			    Draw_Missile(cr, Missile_X_Pos[i],Missile_Y_Pos[i],
//			      Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);
//							/* erase old */
//					clear_prev_path(cr,PrevMissile);
//					Missile_Should_Clean[i] = 1;
			    Missile_X_Pos[i]=Missile_X_Pos[i]+Missile_X_Speed[i];
			    Missile_Y_Pos[i]=Missile_Y_Pos[i]+Missile_Y_Speed[i];
//			    Draw_Missile(cr, Missile_X_Pos[i],Missile_Y_Pos[i],
//			     Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);
//					Missile_Should_Update[i] = 1;
							  /* draw new */
//					stroke_in_clip(cr);

			   }
			 }
      } /* end switch */

		/******** handle new missile **************/

 if(New_Missile_Flag && (Missiles_Counter<5))
  do {
      New_Missile_Flag=OFF;

      Missiles_Counter++;
      for(i=0;i<MAX_NO_OF_MISSILES;i++)
	 			if(Missile_Flag[i]==DEAD) break; /* from for-loop */
      	Missile_Flag[i]=ALIVE;
      	Fire_Missile(i);
//	 	 			Missile_Stock--;
//	  			Update_Shots(cr);

   } while(OFF); /* to enable the break command */
}


//
//int main()
//{
//	printf("Yo man! \n");
//}
//
