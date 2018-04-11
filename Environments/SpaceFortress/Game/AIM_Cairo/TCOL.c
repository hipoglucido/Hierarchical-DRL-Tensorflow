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

//extern Get_Ship_Input(); // not used or something
/* Uncomment (for HM.c)
extern void Update_Ship_Dynamics();
extern void Update_Ship_Display();
extern void Move_Ship();
extern void Fire_Missile();
extern void Handle_Missile(cairo_t *cr);
extern void Generate_Mine();
extern void Move_Mine();
extern void Handle_Mine(cairo_t *cr);
extern void Fire_Shell();
extern void Handle_Shell(cairo_t *cr);
extern void Handle_Fortress(); */

// ------- Uncomment (RE) ------------ //
//extern Update_Vulner();
//extern Update_Velocity();
//extern Update_Control();
//extern Update_Points();
//extern Reset_Screen();
//extern Mydelay(unsigned Timedelay);
// --------------------------------- //


// Added in header
//char *Small_Expl_Buffer;
//int Data_Update_Counter=20;
//int Last_Center_Dist;
//int Wrap_Around_Flag=OFF;
/* int Last_Missile_Hit=0;  to measure interval between two consecutive
			    hits of the fortress */
/*int Ship_Killings_Counter=0; */


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

  int i;
  int Handle_Missile_Flag;

  Handle_Missile_Flag=OFF;

  for(i=0;i<6;i++)   /* for all  possible missiles */
    {                  /* check against mine only */
  if(Mine_Flag==ALIVE)
    if(Missile_Flag[i]==ALIVE)
      if(Check_Collision(Missile_X_Pos[i],Missile_Y_Pos[i],
  	    Mine_X_Pos,Mine_Y_Pos,COLLISION_DIST) )
        {
    Missile_Flag[i]=KILL;
    Handle_Missile_Flag=ON;

    Mine_Flag=KILL;
    Score=1.0;
    Handle_Mine();
      } /* end missile vs. mine for aiming test */
    }
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
