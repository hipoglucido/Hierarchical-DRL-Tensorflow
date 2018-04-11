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

int Data_Update_Counter=20;
int Last_Center_Dist;

#define FIELD_OF_VISION 0
#define sign(x) ((x) < 0 ? -1 : 1)


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

#define DEG2RAD(DEG) (DEG * (M_PI / 180.0))


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

/*                ******************************
_________________________________________________
___  __ \__  ____/__  /___  ____/__  __/__  ____/
__  / / /_  __/  __  / __  __/  __  /  __  __/
_  /_/ /_  /___  _  /___  /___  _  /   _  /___
/_____/ /_____/  /_____/_____/  /_/    /_____/

_____________  ________________
___  __/__  / / /___  _/_  ___/
__  /  __  /_/ / __  / _____ \
_  /   _  __  / __/ /  ____/ /
/_/    /_/ /_/  /___/  /____/
 __   __   __   __   __   __   __   __   __   __   __   __   __   __
|  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  |
|  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  |
|  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  | |  |
|__| |__| |__| |__| |__| |__| |__| |__| |__| |__| |__| |__| |__| |__|
(__) (__) (__) (__) (__) (__) (__) (__) (__) (__) (__) (__) (__) (__)


* *********************************       */

void cairo_line2(cairo_t *cr, int x1, int y1, int x2, int y2) {
	cairo_move_to(cr, x1, y1);
	cairo_rel_line_to(cr, x2, y2);
}

int get_best_move() {

	// Calculate the x and y of the ship's direction unit length vector
	float ship_x = sin(DEG2RAD(Ship_Headings));
	float ship_y = -1*cos(DEG2RAD(Ship_Headings));

	float square_ship_x = (Square_X + ((float)SQUARE_WIDTH)*0.5) - Ship_X_Pos;
	float square_ship_y = (Square_Y + ((float)SQUARE_WIDTH)*0.5) - Ship_Y_Pos;

	// calculate vector length
	float l = sqrt((square_ship_x*square_ship_x) + (square_ship_y*square_ship_y));
	// cos(x) = v (dot). w / ||v|| * ||w||
	float ship_square_angle = (ship_x*square_ship_x + ship_y*square_ship_y) / l;

	// Get the minimal angle threshold from the distance
	float angle_thresh = l > 72 ? 0.98 : 0.73;

	if(ship_square_angle < angle_thresh) {
		float ss_dir = (atan2(square_ship_y,square_ship_x) * 180.0)/M_PI + 180.0;
		float s_dir = (Ship_Headings+90)%360; // change coordinate thingy
		return ss_dir > s_dir ? 2 : 0;
	}
	else { // Facing the right direction
		return 1; // 1 equals forward in gym
	}

}

float direction_score() {
	// Calculate the x and y of the ship's direction unit length vector
	float ship_x = sin(DEG2RAD(Ship_Headings));
	float ship_y = -1*cos(DEG2RAD(Ship_Headings));
//	float ship_x2 = cos(DEG2RAD(SHIP_HEADINGS + FIELD_OF_VISION));
//	float ship_y2 = sin(DEG2RAD(SHIP_HEADINGS + FIELD_OF_VISION));



	float square_ship_x = (Square_X + ((float)SQUARE_WIDTH)*0.5) - Ship_X_Pos;
	float square_ship_y = (Square_Y + ((float)SQUARE_WIDTH)*0.5) - Ship_Y_Pos;

	// calculate vector length
	float l = sqrt((square_ship_x*square_ship_x) + (square_ship_y*square_ship_y));
	// cos(x) = v (dot). w / ||v|| * ||w||
	float ship_square_angle = (ship_x*square_ship_x + ship_y*square_ship_y) / l;

	#ifdef DEBUG
//	printf("SquareX %d SquareY %d square_ship_x %f square_ship_y %f\nship_x %f ship_y %f\n", Square_X, Square_Y, square_ship_x, square_ship_y, ship_x, ship_y);
//	printf("Angle %f - l: %f\n", ship_square_angle, l);

//	printf("SquareX %d SquareY %d square_ship_x %f square_ship_y %f\nship_x %f ship_y %f\n", Square_X, Square_Y, square_ship_x, square_ship_y, ship_x, ship_y);
	// printf("Angle %f - l: %f\n", ship_square_angle, l);
	cairo_line2(SF_rgb_context, Ship_X_Pos, Ship_Y_Pos, 24.0*ship_x, 24.0*ship_y);
	cairo_set_source_rgb(SF_rgb_context, 255.0/255.0, 71.0/255.0, 254.0/255.0); // pink
	cairo_stroke(SF_rgb_context);
	cairo_line2(SF_rgb_context, Ship_X_Pos, Ship_Y_Pos, square_ship_x, square_ship_y);
	cairo_set_source_rgb(SF_rgb_context, 16.0/255.0, 201.0/255,255.0/255.0); // blue
	cairo_stroke(SF_rgb_context);
	#endif

	float angle_thresh = l > 72 ? 0.93 : 0.67;

	// Source: http://stackoverflow.com/a/293052/1657933
	// Calc. if the inverted ship_dir vector  intersects the square
	// because we're dealing with vectors, we can simplify
	// this to (x1, y1) = (0, 0)
	// (the vector is [x2, y2])
	// F(x,y) = (y2-y1)x + (x1-x2)y + (x2*y1-x1*y2) =
	// F(x,y) = y2*x + x2*y



	// Check if the ship's direction is facing the square
//	#ifdef NO_WRAP
	if(ship_square_angle < angle_thresh) {
		return 0.0;
	}
	else { // Facing the right direction
		return 0.015;
	}
//	#else
//	if(fabs(ship_square_angle) > angle_thresh) {
//		float inv_dir_x = -1*ship_x;
//		float inv_sq_x = -1*square_ship_x;
//
//		// Compare the first line drawn from the  horizonatal alligned PU
//		// PU horizontal offset + square border location
//		int PUx = sign(inv_dir_x);
////		int PUx_offset = ship_square_angle < 0 ? 0 : PUx * MaxX;
//		int PUx_offset = PUx * MaxX;
//		int PUH_SquareX = PUx_offset + Square_X;
////		printf("Og_Idx: (%d,%d) ",  sign(ship_x), sign(ship_y));
//
////		printf("PU_Idx: (%d,%d) ",  sign(inv_dir_x), sign(-1*ship_y));
//
////		printf("Pu idx: (%d, %d) Og: (%d, %d)-", sign(inv_dir_x), sign(-1*ship_y), sign(ship_x), sign(ship_y));
//
//		// The inverted direction is a horizontal hit
//		if(PUx==-1) {
//			if((Ship_X_Pos+inv_sq_x) < (PUH_SquareX + SQUARE_WIDTH)) {
////				printf("(1) ");
////				printf("PUH_SquareX: %d Ship_X_Pos: %d inv_sq_x: %f \n", PUH_SquareX, Ship_X_Pos, inv_sq_x);
////				printf("Inverted horizontal hit found\n");
//				return 0.0;
//			}
//		}
//		else { // == 1
//			if(Ship_X_Pos+inv_sq_x > PUH_SquareX) {
////				printf("(2) ");
////				printf("PUH_SquareX: %d Ship_X_Pos: %d inv_sq_x: %f \n", PUH_SquareX, Ship_X_Pos, inv_sq_x);
////				printf("Inverted horizontal hit found\n");
//				return 0.0;
//			}
//		}
//
//		float inv_dir_y = -1*ship_y;
//		float inv_sq_y = -1*square_ship_y;
//
//		int PUy = sign(inv_dir_y);
//		int PUy_offset = sign(inv_dir_y) * MaxY;
//		int PUV_SquareY = PUy_offset + Square_Y;
//
//		// The inverted direction is a vertical hit
//		if(PUy==-1) {
//			if((Ship_Y_Pos+inv_sq_y) < (PUV_SquareY + SQUARE_WIDTH)) {
////					printf("(1) ");
////					printf("PUV_SquareY: %d Ship_Y_Pos: %d inv_sq_y: %f \n", PUV_SquareY, Ship_Y_Pos, inv_sq_y);
////					printf("Inverted vertical hit found\n");
//					return 0.0;
//			}
//		}
//		else { // == 1
//			if((Ship_Y_Pos+inv_sq_y) > PUV_SquareY) {
////					printf("(2) ");
////					printf("PUV_SquareY: %d Ship_Y_Pos: %d inv_sq_y: %f \n", PUV_SquareY, Ship_Y_Pos, inv_sq_y);
////					printf("Inverted vertical hit found\n");
//					return 0.0;
//			}
//		}
//		// Cool nothing else is faster
//		if(ship_square_angle < 0) {
////			printf("PUH_SquareX: %d Ship_X_Pos: %d inv_sq_x: %f \n", PUH_SquareX, Ship_X_Pos, inv_sq_x);
////			printf("PUV_SquareY: %d Ship_Y_Pos: %d inv_sq_y: %f \n", PUV_SquareY, Ship_Y_Pos, inv_sq_y);
//			return 0.0;
//		}

////		printf("Cool direciton baby 😎\n");
//		return 3.0;
//	}
//	else {
//		return 0.0;
//	}
//	#endif
}

// The score function is equal f(t) =  -(x*t^2) + b, with x = (b/m^2), and
// m some constant c, where c represents that maximum amount of time steps
// you get for reaching the square. t is the time it took you to reach the square
// b is the distance from the previous square location, and x is the width of
// of the graph that should equal zero when t = m
float score_function()
{
	float m = (float)MAX_SQUARE_STEPS;
	float m_sqr = m*m;
	float x = Prev_Square_Dist/(m_sqr);
	float t = MAX_SQUARE_STEPS - (float) Square_Step;
	float score = -1.0*(x*(t*t)) + Prev_Square_Dist;
	return score + 1.0; // Add one because we always should get some points
}

void Test_Collisions()
{
	// Test square..
	// ... if hit, update points, if not

	if( ((Ship_X_Pos > (Square_X-COLLISION_DIST)) && (Ship_X_Pos < (Square_X + SQUARE_WIDTH +
	COLLISION_DIST)))
	&&
	((Ship_Y_Pos > (Square_Y-COLLISION_DIST)) && (Ship_Y_Pos < (Square_Y + SQUARE_HEIGHT +
	COLLISION_DIST)))
	)
	{
		#ifdef DEBUG
			#ifdef __APPLE__
				int text_flag = randrange(0, 3);
				switch(text_flag)
				{
					case 0:
						system("say 'Woohoo!'&");
						break;
					case 1:
						system("say 'Good job!'&");
						break;
					case 2:
						system("say 'Keep going!'&");
						break;
					case 3:
						system("say 'Yes!'&");
						break;
				}
			#else
				system("tput bel &");
				printf("Got square! 🙌\n");
			#endif
		#endif
		Square_Flag = KILL;
		Score = 1;
	}
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

       }
//     Update_Control(cr);
//     Update_Points(cr);

   } /* if data-update-counter */
}
