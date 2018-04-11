/* test collision and generate effects   */

#include <stdio.h>
#include <stdlib.h>
//#include <graphics.h>
//#include <process.h>
//#include <process.h>
//#include <bios.h>
//#include <alloc.h>
//#include <dos.h>
#include <time.h>
#include <math.h>
#include <cairo.h>
#include <unistd.h>

//#include "DE.h"
#include "HM.h"
#include "RS.h"
#include "TCOL.h"


//#include "myconst.h"
//#include "myext.h"

// Addded
//#include "HM.h"


char *Small_Expl_Buffer;
int Data_Update_Counter=20;
int Last_Center_Dist;
int Wrap_Around_Flag=OFF;



// Okay to drop animation?
void Gen_Explosion(int X_Pos,int Y_Pos,int Radius)
{
//  int i,j;
//  int iarc;
//	g_signal_connect(G_OBJECT(darea), "draw", G_CALLBACK(on_draw_event), NULL);
//	cairo_reset_clip(cr);
//	cairo_rectangle(cr, 0, 0, 320, 240);
//	cairo_paint(cr);
/* -- unused --
  int X_dot,Y_dot;
  int svcolor;
  int Last_Pitch; */
  Effect_Flag=ON;
	Explosion_Flag=1;
//  svccd olor=getcolor();

	ExpRadius = Radius;
	ExpX = X_Pos;
	ExpY = Y_Pos;

}

void Zero_Vulner_Sound()
{
//  sound(600);
//  Sound_Flag=4;
//  return(0);
}

// -- These kind of animation functions maybe should get some sort of special treatment
// within the step function as they redraw the ship with a delay multiple times --
void Jitter_Ship()
{
//  int Jitter_Headings;
//  int Jitter_X_Pos,Jitter_Y_Pos;
//  int i;

  Effect_Flag=ON;
	Jitter_Flag=1;

}

void Reset_All_Missiles()
{
  int i;

  for (i=0;i<MAX_NO_OF_MISSILES;i++)
      if(Missile_Flag[i]==ALIVE)  Missile_Flag[i]=KILL;
  Handle_Missile();
}

int Check_Collision(float First_X,float First_Y,float Second_X,
		    float Second_Y, int Crash_Distance)
{
  int dist;

  dist=fabs(sqrt(pow(First_X-Second_X,2)+
		 pow(First_Y-Second_Y,2)  ));
  if(dist<Crash_Distance) return(1);
		     else
		     return(0);
}

void Test_Collisions()
{
  int breakflag;
  int i;
  int Handle_Missile_Flag;


  Handle_Missile_Flag=OFF;
  breakflag=OFF;

	/******* mine vs. ship collision ***********/
  // int goodshot;
//   if(Mine_Flag==ALIVE)
//     if(Check_Collision(Ship_X_Pos,Ship_Y_Pos,Mine_X_Pos,Mine_Y_Pos, COLLISION_DIST) )
//       {
// 	Ship_Killings_Counter++;
// 	Ship_Damaged_By_Mines++;
// 	breakflag=ON; 	/* no use to check others */
// 	if(Ship_Killings_Counter>=4)
// 	   {
// 	     Points=Points-100;
// 	     Gen_Explosion(Ship_X_Pos,Ship_Y_Pos,80);
// //			 Terminal_State = 1;
// 				Terminal_State_Flag = FORT_WON;
// 	     Ship_Killings_Counter=0;
// //	     Reset_Screen(cr);
// 	   }
// 	 else
// 	   {
// 	     Points=Points-50;
// 	     Mine_Flag=KILL;
// 	     Handle_Mine(); 		/* kill mine */
// 	     if(Shell_Flag==ALIVE)
// 	     {
// 		  	Shell_Flag=KILL;      /* kill shell */
// 		  	Handle_Shell();
// 	     }
// 	     Reset_All_Missiles();    	/* kill all missiles */
// //	     Gen_Snap_Effect();
// 	     Jitter_Ship();		/* leaves ship on screen */
// 	   }
//       }  /* end ship vs. mine collision */


		/******** shell vs. ship collision *********/

  if((Shell_Flag==ALIVE) && (!breakflag) )
	 if(Check_Collision(Ship_X_Pos,Ship_Y_Pos,
			    Shell_X_Pos,Shell_Y_Pos,COLLISION_DIST) )
	   {
	     Ship_Killings_Counter++;
	     Ship_Damaged_By_Fortress++;
	     breakflag=ON;
       Score = -1.0; // Breakflag is when something is already hit?
	     if(Ship_Damaged_By_Fortress>=2) // was 4
	     {
	     	Shell_Flag=KILL;
		//  Points=Points-100;
		    // Gen_Explosion(Ship_X_Pos,Ship_Y_Pos,80);
			  Terminal_State = 1;
			  Terminal_State_Flag = FORT_WON;
		    Ship_Damaged_By_Fortress=0;
   	      	#ifdef DEBUG
	      	system("say \"Game over :(.\"&");
	      	reset_sf();
	      	#endif
//		 Reset_Screen();
	     }
	     else
	     {
   	      	#ifdef DEBUG
	      	system("say \"Ouch!\"&");
	      	#endif
  		//  Points=Points-50;
  		 Shell_Flag=KILL;        /* kill shell */
  		 Handle_Shell(); // Uncomment when done
  		 if(Mine_Flag==ALIVE)    /* kill  mine  */
  		   {
  		     Mine_Flag=KILL;
  		     Handle_Mine(); /* erase mine and reset counters */
  		   }
  		 Reset_All_Missiles();
  		//  Jitter_Ship();     	/* leaves ship on screen */
	     }
	   }

  if(!breakflag)
  for(i=0;i<MAX_NO_OF_MISSILES;i++)   /* for all  possible missiles */
  {                  /* check against mine and fortress */
    // if(Mine_Flag==ALIVE)
    //   if(Missile_Flag[i]==ALIVE)

		/***** check missile vs. mine ********/

// 	 if(Check_Collision(Missile_X_Pos[i],Missile_Y_Pos[i],
// 			    Mine_X_Pos,Mine_Y_Pos,COLLISION_DIST) )
//
// 	   {
// 	     Missile_Flag[i]=KILL;
// 	     Handle_Missile_Flag=ON;
// 	     goodshot=OFF;
// 	     if((Missile_Type==VS_FRIEND)&&(Mine_Type==FRIEND))
// 	       {
// 		 goodshot=ON;
// 		 Points=Points+20;
// 		 Vulner_Counter++;
// //		 Update_Vulner(cr);
//
// 	       }
// 	     else
// 	     if((Missile_Type==VS_FOE)&&(Mine_Type==FOE))
// 	       {
// 		 goodshot=ON;
// 		 Points=Points+30;
// 	       }
// 	     if(goodshot)
// 	       {
// 		 goodshot=OFF; /* redundant */
// 		 Gen_Snap_Effect();
// 		 Mine_Flag=KILL;
// 		 Handle_Mine();
// 	       }
// 	  } /* end missile vs. mine */

		/******** misile vs. fortress *********/

//  if(!Missile_Vs_Mine_Only) // was on, torn for mines ❗️
//    if(Missile_Flag[i]==ALIVE) // same as above
	 if(Check_Collision(Missile_X_Pos[i],Missile_Y_Pos[i],
			    MaxX/2,MaxY/2,COLLISION_DIST) )
	 {
	    // New: (copied from shell vs. ship)
	     Missile_Flag[i]=KILL;
//		printf("Loop_Counter: %d Last_Missile_Hit: %d\n", Loop_Counter, Last_Missile_Hit);
	    if(Loop_Counter-Last_Missile_Hit>10)  /* 6 loops ...*/
		{
	     Handle_Missile_Flag=ON;
		
	      Score = 1.0;
	      if(Vulner_Counter > 1) // was >= 4 (DEATH)
	      {
	      	#ifdef DEBUG
	      	system("say \"You won!\"&");
	      	reset_sf();
	      	#endif
	       Terminal_State = 1;
	       Terminal_State_Flag = SHIP_WON;
	       Vulner_Counter = 0;
//	       Handle_Missile_Flag=OFF;
   	       break;
	      }
	      else {
	      	#ifdef DEBUG
	      	system("say \"Gotcha!\"&");
	      	#endif
		      Vulner_Counter++;
	     //  Points=Points-50;
	     //  Jitter_Ship();     	/* leaves ship on screen */
	      }
    	}
	    else {
	      	#ifdef DEBUG
	      	system("say \"Too fast.\"&");
	      	#endif
	    }
	    Last_Missile_Hit=Loop_Counter;
  }
  

    // Old:
	  //  Missile_Flag[i]=KILL;
	  //  Handle_Missile_Flag=ON;
// 	   if(Missile_Type==VS_FRIEND)
// 	    if(Vulner_Counter>=11) /* fortress destruction */ // was 11
// 	     if(Loop_Counter-Last_Missile_Hit<6)  /* 6 loops ...*/
// 	      {
// 				Fortress_Destroyed++;
// 				Points=Points+204; /* including the last missile */ // was 104
// 				Vulner_Counter=0;
//          if(Bonus_Granted)
// 		     {
// //			Write_Bonus_Message();     /* erase bonus message */
// 					Bonus_Granted=OFF;
// 		     }
// 		// Gen_Explosion(Missile_X_Pos[i],Missile_Y_Pos[i],120);
// 		Terminal_State_Flag = SHIP_WON;
// //		Terminal_State = 1;
// //		Reset_Screen(cr)
// 		Handle_Missile_Flag=OFF;
// 		Last_Missile_Hit=Loop_Counter;
// 		break;  /* no more missiles checks */
// 	      }
// 	     else /*  >=6 you're too slow my friend.. */
// 	      {
// 		Points=Points+50; /* is this correct */ // was 4
// 		Vulner_Counter++;
// //		Update_Vulner(cr);
// 		Last_Missile_Hit=Loop_Counter;
// 	      }
// 	    else /* Vulner_Counter<11 */
// 			{
// 				if(Loop_Counter-Last_Missile_Hit>=6)
// 				{
// 				  Vulner_Counter++;
// //					Update_Vulner(cr);
// 				  Points=Points+50;
// 				  Last_Missile_Hit=Loop_Counter;
// 				}
// 				else /* double strike before it's OK */
// 				{
// 				  Vulner_Counter=0; /* for speeeding, ha ha ha .... */
// //					Update_Vulner(cr);
// 				  Last_Missile_Hit=Loop_Counter;
// //				  Zero_Vulner_Sound();
// 				}
// 			}
// 	 } /* missile vs. fortress */
  } /* end for missile do-loop */
  if(Handle_Missile_Flag) Handle_Missile(); /* KILL them all */
}

void Accumulate_Data()
{
  float shipvel;
  int shipcenterdist;

  if(--Data_Update_Counter<=0)
   {
     Data_Update_Counter=20;

	/* update Velocity */
     shipvel=sqrt(pow(Ship_X_Speed,2)+pow(Ship_Y_Speed,2));
		if(shipvel<SHIP_GOOD_VELOCITY)
		{
 			Velocity=Velocity+7;
//			Update_Velocity(cr); // --- UNCOMMENT --- //
		}

	/* update Control */

     shipcenterdist=sqrt(pow(Ship_X_Pos-MaxX/2,2)+
			 pow(Ship_Y_Pos-MaxY/2,2));

     if((shipcenterdist<SMALL_HEXAGONE_SIZE_FACTOR*MaxX)&&
	(Last_Center_Dist>SMALL_HEXAGONE_SIZE_FACTOR*MaxX))
	 Control=Control-5;
     else
     if(shipcenterdist<BIG_HEXAGONE_SIZE_FACTOR*MaxX) Control=Control+7;
     else
				     Control=Control+3;
     Last_Center_Dist=shipcenterdist;

		/* if Wrap_Around  */

     if(Wrap_Around_Flag)
       {
	 Control=Control-35;
	 Wrap_Around_Flag=OFF;
       }
//     Update_Control(cr);
//     Update_Points(cr);

   } /* if data-update-counter */
}

//int main()
//{
//	start_drawing();
//	Ship_Y_Pos +=50;
//	Gen_Explosion(SF_canvas, Ship_X_Pos, Ship_Y_Pos, 80);
//	Draw_Ship(SF_canvas, Ship_X_Pos, Ship_Y_Pos, 90,SHIP_SIZE_FACTOR*MaxX);
//	double x1;
//	double y1;
//	double x2;
//	double y2;
//	cairo_path_extents(SF_canvas,&x1,&y1,&x2,&y2);
//	cairo_stroke(SF_canvas);
//
//	cairo_set_source_rgb(SF_canvas,1,0,0);
//	cairo_rectangle(SF_canvas, x1, y1, x2-x1, y2-y1);
//	cairo_stroke(SF_canvas);
//	cairo_surface_write_to_png(surface, "exp.png");
//	stop_drawing();
////	start_drawing(); ////	Gen_Explosion(SF_canvas, MaxX/2, MaxY/2, 120);
////	stop_drawing();
//}
