/* ******************   Handle_Mine and all others elements   ***************** */
#include <stdio.h>
#include <stdlib.h>
#include <graphics.h>
#include <process.h>
#include <bios.h>
#include <alloc.h>
#include <dos.h>
#include <time.h>
#include <math.h>

#include "myconst.h"
#include "myext.h"

int Mine_Alive_Counter=0;
int Mine_Dead_Counter=0;
int Missile_Delay_Counter=0;

char Char_Set[10][1]={"Y","M","P","B","Q","K","C","W","R","Z"};
char Tmp_Char_Set[10][1];

char Foe_Menu[3][1];
char Friend_Menu[3][1];
char Mine_Indicator;
mine_type Mine_Type;

int Get_Random_Index(int vec[])
{
  int k;

  do
     { k=random(10); }
  while (vec[k]==-1);

  vec[k]=-1;
  return(k);
}

int Select_Mine_Menus()
{
  int vec[10];
  int i,ri;

  for(i=0;i<10;i++) vec[i]=0;
  for (i=0;i<3;i++)
      {
	ri=Get_Random_Index(vec);
	Friend_Menu[i][0]= Char_Set[ri][0];
	ri=Get_Random_Index(vec);
	Foe_Menu[i][0]= Char_Set[ri][0];
      }
  return(0);
}

int Update_Ship_Dynamics()
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
  if(Game_Type==SPACE_FORTRESS)
  {
  if (Accel_Input!=0)
     {
       Ship_X_Speed=Ship_X_Speed+0.65*Ship_Accel*Fsin(Ship_Headings);
       Ship_Y_Speed=Ship_Y_Speed-0.65*Ship_Accel*Fcos(Ship_Headings);
       Accel_Input=0; 	/* reset input */

       /* assure it does not exceed MAXspeed */

       if(fabs(Ship_X_Speed)>Ship_Max_Speed)
		 if(Ship_X_Speed<0) Ship_X_Speed=-Ship_Max_Speed;
				    else
				    Ship_X_Speed=Ship_Max_Speed;
       if(fabs(Ship_Y_Speed)>Ship_Max_Speed)
		 if(Ship_Y_Speed<0) Ship_Y_Speed=-Ship_Max_Speed;
				    else
				    Ship_Y_Speed=Ship_Max_Speed;

     }  /* end accel_input */
		/* now update ship position */

    if ((Ship_X_Speed!=0.0)||(Ship_Y_Speed!=0.0))
     {
       Ship_Display_Update=1; /* ship moves */

       Ship_X_Pos=Ship_X_Pos+Ship_X_Speed;
       Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
			/* check if crossed screen boundary */
       if(Ship_X_Pos<0) { Ship_X_Pos=MaxX;
			  Wrap_Around_Flag=ON; }
       if(Ship_X_Pos>MaxX) { Ship_X_Pos=0;
				     Wrap_Around_Flag=ON; }
       if(Ship_Y_Pos<0) { Ship_Y_Pos=MaxY;
			  Wrap_Around_Flag=ON; }
       if(Ship_Y_Pos>MaxY) { Ship_Y_Pos=0;
				     Wrap_Around_Flag=ON; }
			/* check if bumped into the fortress */
       if(sqrt(pow(Ship_X_Pos-MaxX/2,2)+
	       pow(Ship_Y_Pos-MaxY/2,2) ) < (COLLISION_DIST) )
	 {
	   Ship_X_Speed=-Ship_X_Speed;		/* reverse direction */
	   Ship_Y_Speed=-Ship_Y_Speed;
	   Ship_X_Pos=Ship_X_Pos+Ship_X_Speed; /* move ship out of range */
	   Ship_Y_Pos=Ship_Y_Pos+Ship_Y_Speed;
	 }
     } /* end ship is moving */
  } /* end SPACE_FORTRESS */
  return(0);
}

int Update_Ship_Display()
{
		/* erase ship in old location */
  Draw_Ship(Ship_X_Old_Pos,Ship_Y_Old_Pos,Ship_Old_Headings,SHIP_SIZE_FACTOR
							     *MaxX);
		/* draw ship in new location */
  Draw_Ship(Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);
  return(0);
}

int Move_Ship()
{
  Update_Ship_Dynamics();
  if(Ship_Display_Update)  Update_Ship_Display();
  return(0);
}

int Fire_Shell()
{
  Shell_X_Pos=MaxX/2+0.5*SMALL_HEXAGONE_SIZE_FACTOR*MaxX*Fsin(Fort_Headings);
  Shell_Y_Pos=MaxY/2-0.5*SMALL_HEXAGONE_SIZE_FACTOR*Fcos(Fort_Headings);
  Shell_Headings=Fort_Headings;
  Shell_X_Speed=SHELL_SPEED*Fsin(Shell_Headings);
  Shell_Y_Speed=-SHELL_SPEED*Fcos(Shell_Headings);
  Draw_Shell(Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
				SHELL_SIZE_FACTOR*MaxX);  /* first time */
  sound(800);
  Sound_Flag=6;
  return(0);
}

int Handle_Fortress()
{
  int dif,nh;

  if( (++Fort_Lock_Counter>FORT_LOCK_INTERVAL)&&(Shell_Flag==DEAD) )
    {
      Fire_Shell();
      Shell_Flag=ALIVE;
      Fort_Lock_Counter=0;
    }

  nh=Find_Headings(MaxX/2,MaxY/2,Ship_X_Pos,Ship_Y_Pos);
  if (abs(Fort_Headings-nh)>10)
     {
       Draw_Fort(MaxX/2,MaxY/2,Fort_Headings,FORT_SIZE_FACTOR*MaxX);
						/* erase old position */
       Fort_Headings=nh;
       Draw_Fort(MaxX/2,MaxY/2,Fort_Headings,FORT_SIZE_FACTOR*MaxX);
						/* draw new position */
       Fort_Lock_Counter=0;  /* reset firing counter */
     }
  return(0);
}

int Handle_Speed_Score()
{
  int interval;
  int dts;

  if(Game_Type==SPACE_FORTRESS)
  {
  dts=0;
  if(Mine_Type==FOE)
    if(Missile_Type==VS_FOE) /* successful double press */
      {
	interval=t0-t2; /* time interval from missile */
			/* appearance to end of successful */
			/* double press interval      */
	if(interval<=1000) dts=40;
	else
	if(interval<=2000) dts=30;
	else
	if(interval<=3000) dts=20;
	else
	if(interval<=4000) dts=10;
	else
	if(interval<=5000) dts=0;
	else
	if(interval<=6000) dts=-10;
	else
	if(interval<=7000) dts=-20;
	else
	if(interval<=8000) dts=-30;
	else
	if(interval<=9000) dts=-40;
	else
	dts=-50;
     }
     else /* no successful double-press */
	dts=-50; /* this is "unfair" */
   Speed=Speed+dts;
  } /* end if(SPACE_FORTRESS) */

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
  Update_Speed();
  if(Game_Type==AIMING_TEST)
    {
      Score=Mines+Speed;
      Update_Score();
    }
  return(0);
}

int Clear_Mine_Type()
{
  int x,y;

  setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);
  x=IFF_X; y=Data_Line;
  putimage(x,y,buffer1,COPY_PUT); /* erase garbage */
  setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);
  return(0);
}

int Show_Mine_Type(char Minetype)
{
  int svcolor;
  int x,y;

  svcolor=getcolor();
  if((Mine_Type==FRIEND && Missile_Type==VS_FRIEND) || (Mine_Type==FOE && Missile_Type==VS_FOE)) {
    setcolor(GREEN);
  } else if(Missile_Type==WASTED) {
    setcolor(RED);
  } else {
    setcolor(LIGHTRED);
  }
  setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);
  x=IFF_X; y=Data_Line;
  putimage(x,y,buffer1,COPY_PUT); /* erase garbage */
  gprintf(&x,&y,"%c",Minetype);
  setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);
  setcolor(svcolor); /* restore previous color */
  return(0);
}

Reset_Mine_Headings()
{
   Mine_Headings=Find_Headings(Mine_X_Pos,Mine_Y_Pos,Ship_X_Pos,
					     Ship_Y_Pos);
   Mine_Course_Count=MINE_COURSE_INTERVAL;
   Mine_X_Speed=Mine_Speed*Fsin(Mine_Headings);
   Mine_Y_Speed=-Mine_Speed*Fcos(Mine_Headings);
   return(0);
}

int Generate_Mine()
{
  int a;
  do
  {
    Mine_X_Pos=random(MaxX);
    Mine_Y_Pos=random(MaxY);
    a=sqrt(pow(Mine_X_Pos-Ship_X_Pos,2)+pow(Mine_Y_Pos-Ship_Y_Pos,2) );
  } while(a < 0.5*MaxX );  /* repeat until distance exceeds min. */

  Reset_Mine_Headings();
  Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);  /* draw mine first time */

  if(random(2)) Mine_Type=FRIEND;
  else
    {
      Mine_Type=FOE;
      t0=Time_Counter; /* when "a mine is born .."? */
    }

  if (Mine_Type==FRIEND) Mine_Indicator=Friend_Menu[random(3)][0];
  else                   Mine_Indicator=Foe_Menu[random(3)][0];
  Show_Mine_Type(Mine_Indicator);
  return(0);

}

int Move_Mine()
{
    Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX); /* erase mine */

    Mine_X_Pos=Mine_X_Pos+Mine_X_Speed;      /* update position */
    Mine_Y_Pos=Mine_Y_Pos+Mine_Y_Speed;

    Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);  /* redraw mine */

    if(--Mine_Course_Count<=0)  Reset_Mine_Headings();
    if(   (Mine_X_Pos<0) || (Mine_X_Pos>MaxX)
     || (Mine_Y_Pos<0) || (Mine_Y_Pos>MaxY) )
      Reset_Mine_Headings();
   return(0);
}

int Handle_Mine()
{
 switch(Mine_Flag)
 {
  case KILL  : {
		  Handle_Speed_Score();
		  Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
							/* erase mine */
		  Mine_Flag=DEAD;
		  Mine_Dead_Counter=0;
		  Missile_Type=VS_FRIEND;
		  Missile_Vs_Mine_Only=OFF;
		  Timing_Flag=OFF;
		  Clear_Mine_Type(); /* clear mine type display */
		  Clear_Interval();
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
		  Move_Mine();
		  if(Mine_Alive_Counter++ >= Mine_Live_Loops)
		  Mine_Flag=KILL;
		  if(Mine_Alive_Counter>MISSILE_FORT_TIME_LIMIT)
		    Missile_Vs_Mine_Only=ON;

		 }

 } /* end switch */
 return(0);

}


int Generate_Aim_Mine()
{
    float radius;
    float mine_distance;
    float mine_angle;

    radius=MaxX/2.2;
    mine_angle=random(16)*22.5;
    if(mine_angle>338.0) mine_angle=0.0;
    mine_distance=radius/2+random(2)*radius/2;

    Mine_X_Pos=MaxX/2 + mine_distance*Fsin(mine_angle);
    if(AspectRatio==1.0)
       Mine_Y_Pos=MaxY/2 - mine_distance*Fcos(mine_angle);
    else Mine_Y_Pos=MaxY/2 - mine_distance*Fcos(mine_angle)/GraphSqrFact;
		     /* Y/X square ratio */

    Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
	     /* draw mine */
   return(0);
}

int Handle_Aim_Mine()
{
 switch(Mine_Flag)
 {
  case KILL  : {
		  Handle_Speed_Score();
		  Draw_Mine(Mine_X_Pos,Mine_Y_Pos,MINE_SIZE_FACTOR*MaxX);
							/* erase mine */
		  Mine_Flag=DEAD;
		  Mine_Dead_Counter=0;
		  break;
		}
  case DEAD   : {
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
 return(0);
}


int Handle_Shell()
{
 switch(Shell_Flag)
 {
  case KILL  : {
		  Draw_Shell(Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
			    SHELL_SIZE_FACTOR*MaxX); /* erase shell */
		  Shell_Flag=DEAD;
		  break;
		}

  case ALIVE  : {
		  Draw_Shell(Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
				       SHELL_SIZE_FACTOR*MaxX);
						/* erase shell */
		  Shell_X_Pos=Shell_X_Pos+Shell_X_Speed;
		  Shell_Y_Pos=Shell_Y_Pos+Shell_Y_Speed;
		  if( (Shell_X_Pos<0) || (Shell_X_Pos>MaxX)
		      || (Shell_Y_Pos<0) || (Shell_Y_Pos>MaxY) )
		    Shell_Flag=KILL;  /* kill shell */
		  else
		    Draw_Shell(Shell_X_Pos,Shell_Y_Pos,Shell_Headings,
					SHELL_SIZE_FACTOR*MaxX);
		}
 } /* end switch */
 return(0);
}


int Fire_Missile(int Index)
{
 Missile_Headings[Index]=Ship_Headings;
 Missile_X_Pos[Index]=Ship_X_Pos;
 Missile_Y_Pos[Index]=Ship_Y_Pos;
 Missile_X_Speed[Index]= Missile_Speed*Fsin(Ship_Headings);
 Missile_Y_Speed[Index]=-Missile_Speed*Fcos(Ship_Headings);
 Draw_Missile(Missile_X_Pos[Index],Missile_Y_Pos[Index],
	      Missile_Headings[Index],MISSILE_SIZE_FACTOR*MaxX);
							/* first time */
 sound(1000);
 Sound_Flag=4;
 return(0);
}


int Handle_Missile()
{
 int i;

		/* update all existing missiles */
 for(i=1;i<6;i++)
    if(Missile_Flag[i] != DEAD)
      switch(Missile_Flag[i])
      {
	 case KILL  : {
			Draw_Missile(Missile_X_Pos[i],Missile_Y_Pos[i],
			     Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);  /* erase missile */
			Missile_Flag[i]=DEAD;
			Missiles_Counter--;
			break;
		      }

	 case ALIVE  : {
			 if((Missile_X_Pos[i]<0) || (Missile_X_Pos[i]>MaxX)
			 || (Missile_Y_Pos[i]<0) || (Missile_Y_Pos[i]>MaxY))
			    Missile_Flag[i]=KILL;
			 else
			  {
			    Draw_Missile(Missile_X_Pos[i],Missile_Y_Pos[i],
			      Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);
							/* erase old */
			    Missile_X_Pos[i]=Missile_X_Pos[i]+Missile_X_Speed[i];
			    Missile_Y_Pos[i]=Missile_Y_Pos[i]+Missile_Y_Speed[i];
			    Draw_Missile(Missile_X_Pos[i],Missile_Y_Pos[i],
			     Missile_Headings[i],MISSILE_SIZE_FACTOR*MaxX);
							  /* draw new */
			   }
			 }
      } /* end switch */

		/******** handle new missile **************/

 if(New_Missile_Flag && (Missiles_Counter<5))
  do {
      New_Missile_Flag=OFF;

      if(Game_Type==SPACE_FORTRESS)
	if(Missile_Stock<=0)    /* stock control */
	  if(Missile_Limit_Flag) break;  /* overdraft not allowed */
	  else           Points=Points-3;   /* our low-low interest rates.. */

      Missiles_Counter++;
      for(i=1;i<6;i++)
	 if(Missile_Flag[i]==DEAD) break; /* from for-loop */
      Missile_Flag[i]=ALIVE;
      Fire_Missile(i);
      if(Game_Type==SPACE_FORTRESS)
	{
	  Missile_Stock--;
	  Update_Shots();
	}
   } while(OFF); /* to enable the break command */
 return(0);
}