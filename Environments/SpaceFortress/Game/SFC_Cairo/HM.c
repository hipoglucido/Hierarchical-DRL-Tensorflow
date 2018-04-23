/* ******************   Handle_Mine and all others elements   ***************** */
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


//	printf("X-Speed: %f Y-speed: %f Sum: %f \n", Ship_X_Speed, Ship_Y_Speed, fabs(Ship_X_Speed) + fabs(Ship_Y_Speed));


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
 	} /* end ship is moving */

	#ifdef GRID_MOVEMENT  // If not floaty space movement
	if(Accel_Input == 0)
	{ // Reset speed to get grid like movement
		Ship_X_Speed = 0;
		Ship_Y_Speed = 0;
	}
	#endif
}
#endif


void Handle_Square()
{
	switch(Square_Flag)
	{
		case KILL:
			Generate_Square();
			Square_Flag = ALIVE;
			N_Squares++;

			if (N_Squares > MAX_SQUARES)
			{
			 Terminal_State = 1;
			 N_Squares = 0;
			}
			break;
		case ALIVE:
			Square_Step--;
			if(Square_Step < 0)
			{
				
				Terminal_State = 1;
				Score = -1.0; // There is no observation connected to this atm
				Square_Flag = KILL;
			}
			break;
	}
}



int randrange(int min, int max)
{
	return (rand() % (max + 1 - min)) + min;
}

void Generate_Square()
{
	Square_Step = MAX_SQUARE_STEPS;
	short found_loc = 0;
	int new_square_x;
	int new_square_y;
  	float distance;

	while(!found_loc)
	{

		new_square_x = randrange(0, WINDOW_WIDTH-SQUARE_WIDTH);
		new_square_y = randrange(0, WINDOW_HEIGHT-SQUARE_WIDTH);

		int x_diff = new_square_x - Ship_X_Pos;
		int y_diff = new_square_y - Ship_Y_Pos;

   		 distance = sqrt(pow(x_diff,2) + pow(y_diff,2));

		// + 0.5 square_width for middle point
		if(distance > (2*SQUARE_WIDTH))
		{
			found_loc = 1;
		}

	}

  	Prev_Square_Dist = distance;

	Square_X = new_square_x;
	Square_Y = new_square_y;
}
