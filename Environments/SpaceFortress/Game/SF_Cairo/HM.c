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


// Not needed ?
void Select_Mine_Menus()
{
	int ri, i;
	for(i=0; i < 3; i++) // Populate each array with 3 random characters
	{
		ri=randrange(0, 9);
		Foe_Menu[i] = Char_Set[ri];
		ri=randrange(0, 9);
		Friend_Menu[i] = Char_Set[ri];

	}
}

#ifdef NO_DIRECTION
void Move_Ship()
{
	Ship_X_Old_Pos=Ship_X_Pos;
	Ship_Y_Old_Pos=Ship_Y_Pos;
	if (Rotate_Input!=0) {      // Ship moves left/right
		Ship_X_Pos = SHIP_MAX_SPEED * Rotate_Input + Ship_X_Pos;
		Rotate_Input = 0;
	}
	else if (Accel_Input!=0) {  // Ship moves up/down
		Ship_Y_Pos = SHIP_MAX_SPEED * Accel_Input + Ship_Y_Pos;
		Accel_Input = 0;
	}

	#ifdef NO_WRAP
	if(Ship_X_Pos<0) {
		Ship_X_Pos=0;
	}
    else if(Ship_X_Pos>MaxX) {
    	Ship_X_Pos=MaxX;
	}
    else if(Ship_Y_Pos<0) {
		Ship_Y_Pos=0;
	}
    else if(Ship_Y_Pos>MaxY) {
    	Ship_Y_Pos=MaxY;
	}
	#else
	if(Ship_X_Pos<0) {
		Ship_X_Pos=MaxX;
	}
    else if(Ship_X_Pos>MaxX) {
    	Ship_X_Pos=0;
	}
    else if(Ship_Y_Pos<0) {
		Ship_Y_Pos=MaxY;
	}
    else if(Ship_Y_Pos>MaxY) {
    	Ship_Y_Pos=0;
	}
	#endif
	
	// Check fort bump
    if(sqrt(pow(Ship_X_Pos-MaxX/2,2)+ pow(Ship_Y_Pos-MaxY/2,2) ) < COLLISION_DIST)
	{
	   Ship_X_Speed=-Ship_X_Speed;		/* reverse direction */
	   Ship_Y_Speed=-Ship_Y_Speed;
	   Ship_X_Pos=Ship_X_Pos+Ship_X_Speed; /* move ship out of range */
	   Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
	}

}
#else
void Move_Ship()
{
  Ship_Old_Headings=Ship_Headings;
  Ship_X_Old_Pos=Ship_X_Pos;
  Ship_Y_Old_Pos=Ship_Y_Pos;
//  Ship_Display_Update=0; /* do not refresh if no movement */

  if (Rotate_Input!=0)      /* if ship rotates */
     {
//       Ship_Display_Update=1;  /* at least rotates */
       Ship_Headings=Ship_Headings + Rotate_Input*ROTATE_ANGLE;
       if (Ship_Headings<0) {
         Ship_Headings= 359+Ship_Headings-1;
       }
       else if (Ship_Headings>359) {
         Ship_Headings= Ship_Headings-359-1;
       }
       Rotate_Input=0;        /* reset input */
     }
  else if (Accel_Input!=0)
	{

		#ifdef GRID_MOVEMENT
		// Round because this is rounded down by integer casting, which makes diagonal movement
		// a lot slower compared to vertical or horizontal movement


	    Ship_Y_Speed=round(-4.4*Fcos(Ship_Headings));
   		Ship_X_Speed=round(4.4*Fsin(Ship_Headings));

		#else
	    Ship_X_Speed=Ship_X_Speed+0.65*Ship_Accel*Fsin(Ship_Headings);
	    Ship_Y_Speed=Ship_Y_Speed-0.65*Ship_Accel*Fcos(Ship_Headings);

    	/* assure it does not exceed MAXspeed */

		if(fabsf(Ship_X_Speed)>Ship_Max_Speed)
		{
			Ship_X_Speed = Ship_X_Speed < 0 ? -Ship_Max_Speed : Ship_Max_Speed;
		}
		if(fabsf(Ship_Y_Speed)>Ship_Max_Speed)
		{
			Ship_Y_Speed = Ship_Y_Speed < 0 ? -Ship_Max_Speed : Ship_Max_Speed;
	  }  /* end accel_input */
		#endif
    Accel_Input=0; 	/* reset input */

	}

		/* now update ship position */

  if ((Ship_X_Speed!=0.0)||(Ship_Y_Speed!=0.0))
  {
//       Ship_Display_Update=1; /* ship moves */ // Already set
     Ship_X_Pos=Ship_X_Pos+Ship_X_Speed;
     Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
	/* check if crossed screen boundary */
			#ifdef NO_WRAP
			if(Ship_X_Pos<0 || Ship_X_Pos>MaxX)
			{
				Ship_X_Speed=-Ship_X_Speed;		/* reverse direction */
				Ship_Y_Speed=-Ship_Y_Speed;
				Ship_X_Pos=Ship_X_Pos+Ship_X_Speed; /* move ship out of range */
				Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
			}
			else if(Ship_Y_Pos<0 || Ship_Y_Pos>MaxY)
			{
//				Ship_X_Speed=Ship_X_Speed;		/* reverse direction */
				Ship_Y_Speed=-Ship_Y_Speed;
				Ship_X_Pos=Ship_X_Pos+Ship_X_Speed; /* move ship out of range */
				Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
			}

			#else
	     if(Ship_X_Pos<0) { Ship_X_Pos=MaxX;
		  Wrap_Around_Flag=ON; }
	     if(Ship_X_Pos>MaxX) { Ship_X_Pos=0;
			     Wrap_Around_Flag=ON; }
	     if(Ship_Y_Pos<0) { Ship_Y_Pos=MaxY;
		  Wrap_Around_Flag=ON; }
	     if(Ship_Y_Pos>MaxY) { Ship_Y_Pos=0;
			     Wrap_Around_Flag=ON; }
			#endif
 	}
 	
	// Check fort bump
    if(sqrt(pow(Ship_X_Pos-MaxX/2,2)+ pow(Ship_Y_Pos-MaxY/2,2) ) < COLLISION_DIST )
	{
	   Ship_X_Speed=-Ship_X_Speed;		/* reverse direction */
	   Ship_Y_Speed=-Ship_Y_Speed;
	   Ship_X_Pos=Ship_X_Pos+Ship_X_Speed; /* move ship out of range */
	   Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
	}
 	
	#ifdef GRID_MOVEMENT  // If not floaty space movement
	if(Accel_Input == 0)
	{ // Reset speed to get grid like movement
		Ship_X_Speed = 0;
		Ship_Y_Speed = 0;
	}
	#endif
	
	
}
#endif

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

void Handle_Fortress(cairo_t *cr)
{
//  int dif,nh;
	int nh;

  if( (++Fort_Lock_Counter>FORT_LOCK_INTERVAL)&&(Shell_Flag==DEAD) )
    {
      Fire_Shell();
      Shell_Flag=ALIVE;
      Fort_Lock_Counter=0;
    }

  nh=Find_Headings((MaxX/2.0), (MaxY/2.0), Ship_X_Pos,  Ship_Y_Pos);
  if (abs(Fort_Headings-nh)>10)
  {

		Fort_Headings=nh;
		Fort_Lock_Counter=0;  /* reset firing counter */
  }
}

void Handle_Speed_Score()
{
//  struct timeval tDiff;
	int interval;
  int dts;

  dts=0;
  if(Mine_Type==FOE)
    if(Missile_Type==VS_FOE) /* successful double press */
      {
//			timeval_subtract(&tDiff, &intv_t2, &t0);
			interval= intv_t2-t0; /* time interval from missile */
			/* appearance to end of successful */
			/* double press interval      */
	if(interval<=20) dts=40;
	else
	if(interval<=40) dts=30;
	else
	if(interval<=60) dts=20;
	else
	if(interval<=80) dts=10;
	else
	if(interval<=100) dts=0;
	else
	if(interval<=120) dts=-10;
	else
	if(interval<=140) dts=-20;
	else
	if(interval<=160) dts=-30;
	else
	if(interval<=180) dts=-40;
	else
	dts=-50;
     }
     else /* no successful double-press */
	dts=-50; /* this is "unfair" */
   Speed=Speed+dts;

	/* mine bonus for any type */

   dts=0;
   if(Mine_Alive_Counter>=Mine_Live_Loops) dts=-100;
   else
   if(Mine_Alive_Counter<=20) dts=80;
   else
   if(Mine_Alive_Counter<=40) dts=60;
   else
   if(Mine_Alive_Counter<=60) dts=40;
   else
   if(Mine_Alive_Counter<=80) dts=20;
   else
   if(Mine_Alive_Counter<=100) dts=0;
   else
   if(Mine_Alive_Counter<=120) dts=-10;
   else
   if(Mine_Alive_Counter<=140) dts=-40;
   else
   if(Mine_Alive_Counter<=160) dts=-60;
   else
   if(Mine_Alive_Counter<=180) dts=-80;
   else
   if(Mine_Alive_Counter<=200) dts=-100;

  Speed=Speed+dts;

//  Update_Speed(cr);

}

//
// This function is supposed to erease the last show mine identification type
void Clear_Mine_Type(cairo_t *cr)
{
//  int x,y;
//	Mine_Type_Should_Update = 0;
//	Mine_Type_Should_Clean = 1;
//  setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);
//	cairo_translate(cr, 0, Panel_Y_Start);
//  x=IFF_X; y=Data_Line;
//	cairo_set_source_rgb(cr, 0, 0, 0);
//	cairo_rectangle(cr, x, y, TEXT_WIDTH, TEXT_HEIGHT); //
//	cairo_clip_preserve(cr);
//	cairo_fill(cr);
//	cairo_reset_clip(cr);
//	cairo_translate(cr, 0, -Panel_Y_Start);
//  putimage(x,y,buffer1,COPY_PUT); /* erase garbage */ // Function to draw over  the mine?
//  setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);
}

void Show_Mine_Type(char *Minetype)
{
//  int svcolor;
//  int x,y;
//	Mine_Type_Should_Update = 1;

}

void Reset_Mine_Headings()
{
   Mine_Headings=Find_Headings((double)Mine_X_Pos,(double)Mine_Y_Pos,(double)Ship_X_Pos, (double)Ship_Y_Pos);
   Mine_Course_Count=MINE_COURSE_INTERVAL;
   Mine_X_Speed=Mine_Speed*Fsin(Mine_Headings);
   Mine_Y_Speed=-Mine_Speed*Fcos(Mine_Headings);
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

  Reset_Mine_Headings();

//  Draw_Mine(cr,Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);  /* draw mine first time */
//	Mine_Should_Clean = 1;

		Mine_Type=FRIEND;
//  if(randrange(0,1)) Mine_Type=FRIEND; // was random(2)
//  else
//    {
//      Mine_Type=FOE;
////			gettimeofday(&t0, NULL);
//			t0 = Loop_Counter;
////      t0=clock(); /* when "a mine is born .."? */  // Why and how does it access this?
//    }

  if (Mine_Type==FRIEND)
	{
		Mine_Indicator = (char*)Friend_Menu[randrange(0,2)];
	}
  else
	{
		Mine_Indicator = (char*)Foe_Menu[randrange(0,2)];
	}
	Mine_Char = Mine_Indicator;
//	Show_Mine_Type(Mine_Indicator);
}

void Move_Mine()
{
//		stroke_in_clip(cr);
		// Do something with cairo_save() if approiate
//    Draw_Mine(cr, Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX); /* erase mine */
//		Mine_Should_Clean = 1;


//		printf("Mine_X_Speed: %d \n", Mine_X_Speed);
    Mine_X_Pos=Mine_X_Pos+Mine_X_Speed;      /* update position */
    Mine_Y_Pos=Mine_Y_Pos+Mine_Y_Speed;



    if(--Mine_Course_Count<=0)  Reset_Mine_Headings();
    if(   (Mine_X_Pos<0) || (Mine_X_Pos>MaxX)
     || (Mine_Y_Pos<0) || (Mine_Y_Pos>MaxY) )
		{
      Reset_Mine_Headings();
		}
}

void Handle_Mine()
{
 switch(Mine_Flag)
 {
  case KILL  : {
		  // Handle_Speed_Score();
		  Mine_Flag=DEAD;
		  Mine_Dead_Counter=0;
		  Missile_Type=VS_FRIEND;
		  Missile_Vs_Mine_Only=OFF;
		  Timing_Flag=OFF;
		  break;
		}
  case DEAD   : {
		// Uncomment to spawn mines (and maybe some other stuff)
//		  if(Mine_Dead_Counter++ >= Mine_Wait_Loops)
//		    {
//		      Generate_Mine();
//		      Mine_Flag=ALIVE;
//		      Mine_Alive_Counter=0;
//		    }
		   break;
		}
  case ALIVE  : {
		  Move_Mine();
		  if(Mine_Alive_Counter++ >= Mine_Live_Loops)
			{
		  	Mine_Flag=KILL;
			}
		  if(Mine_Alive_Counter>MISSILE_FORT_TIME_LIMIT)
		    Missile_Vs_Mine_Only=ON;
		 }

 } /* end switch */
}

void Handle_Shell()
{
 switch(Shell_Flag)
 {
  case KILL  : {
//		  Draw_Shell(cr, Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
//			    SHELL_SIZE_FACTOR*MaxX); /* erase shell */
//			clear_prev_path(cr, PrevMine);
//			Shell_Should_Clean = 1;
//			Shell_Should_Update = 1;
		  Shell_Flag=DEAD;
		  break;
		}

  case ALIVE  : {
//		  Draw_Shell(cr, Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
//				       SHELL_SIZE_FACTOR*MaxX);
//			clear_prev_path(cr, PrevMine);
//			Shell_Should_Clean = 1;
		  Shell_X_Pos=Shell_X_Pos+Shell_X_Speed;
		  Shell_Y_Pos=Shell_Y_Pos+Shell_Y_Speed;
		  if( (Shell_X_Pos<0) || (Shell_X_Pos>MaxX)
		      || (Shell_Y_Pos<0) || (Shell_Y_Pos>MaxY) )
			{
		    Shell_Flag=KILL;  /* kill shell */
			}
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
		 	Missile_X_Pos[i] = -1;
		 	Missile_X_Pos[i] = -1;
//			clear_prev_path(cr,PrevMissile);
//			Missile_Should_Clean[i] = 1;
//			Draw_Missile(cr, Missile_X_Pos[i],Missile_Y_Pos[i],
//			     Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);  /* erase missile */
			Missile_Flag[i]=DEAD;
			Missiles_Counter--;
			break;
		      }

	 case ALIVE : {
			 if((Missile_X_Pos[i]<0) || (Missile_X_Pos[i]>MaxX)
			 || (Missile_Y_Pos[i]<0) || (Missile_Y_Pos[i]>MaxY))
				{
			    Missile_Flag[i]=KILL;
					Points=Points-1;
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

	if(Missile_Stock<=0)    /* stock control */
	  if(Missile_Limit_Flag) break;  /* overdraft not allowed */
	  else           Points=Points-0;   /* our low-low interest rates.. */ // POINTS_TAG was -3

      Missiles_Counter++;
      for(i=0;i<MAX_NO_OF_MISSILES;i++)
	 			if(Missile_Flag[i]==DEAD) break; /* from for-loop */
      	Missile_Flag[i]=ALIVE;
      	Fire_Missile(i);
	 	 			Missile_Stock--;
//	  			Update_Shots(cr);

   } while(OFF); /* to enable the break command */
}


//
//int main()
//{
//	printf("Yo man! \n");
//}
//
