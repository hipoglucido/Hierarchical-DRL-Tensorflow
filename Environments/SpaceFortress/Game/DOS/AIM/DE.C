
/* DISPLAY ELEMENTS   6 Feb. 90 18:00
			definitions */
#include <dos.h>
#include <graphics.h>
#include <math.h>

#include "myconst.h"
#include "myext.h"

/*									*/
/*	OPEN GRAPHICS: Initializes the graphics system and reports 	*/
/*	any errors which occured.					*/
/*									*/


void Open_Graphics(void)
{
  int xasp,yasp;


  GraphDriver = DETECT;
		/* Request auto-detection	*/
  /*GraphMode=EGAHI;*/
  initgraph( &GraphDriver, &GraphMode, "" );
  ErrorCode = graphresult();		/* Read result of initialization*/
  if( ErrorCode != grOk ){		/* Error occured during init	*/
    printf(" Graphics System Error: %s\n", grapherrormsg( ErrorCode ) );
    exit( 1 );
  }

  getpalette( &palette );		/* Read the palette from board	*/
  MaxColors = getmaxcolor() + 1;	/* Read maximum number of colors*/

  MaxX = getmaxx();
  MaxY = getmaxy();			/* Read size of screen		*/

  getaspectratio( &xasp, &yasp );	/* read the hardware aspect	*/
  AspectRatio = (double)xasp / (double)yasp; /* Get correction factor	*/
  GraphSqrFact=MaxX*AspectRatio/MaxY;       /* for EGA cases */
  setwritemode(XOR_PUT);
}

void Initialize_Graphics(void)
{
  int Height,OldmaxX;
  int t,t1;
  int x,dx;

  cleardevice();
  Height=textheight("H");		/* Get basic text height */
  OldmaxX=MaxX;
  t1=4*Height;

  Panel_Y_End=MaxY;
  Panel_Y_Start=MaxY-t1+2;
  MaxY_Panel=Panel_Y_End-Panel_Y_Start;

  MaxY=MaxY-t1;
  if(AspectRatio==1.0) /* VGA HI */
    MaxX=MaxY;
    else  /* for all others */
    {
      MaxX=MaxX*AspectRatio;    /********* MaxX and MaxY give a square */
      MaxX=MaxX-t1/AspectRatio;  /******** less two panel lines */
    }
  Xmargin=OldmaxX/2-MaxX/2;
  setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);

  if(Game_Type==SPACE_FORTRESS)
    {          	/* set data value locations for space fortress */
      dx=MaxX/8;
      Points_X=x=2*textwidth("Z");
      x=x+dx; Control_X=x;
      x=x+dx; Velocity_X=x;
      x=x+dx; Vulner_X=x;
      x=x+dx; IFF_X=x;
      x=x+dx; Interval_X=x;
      x=x+dx; Speed_X=x;
      x=x+dx; Shots_X=x;
    }
  else /* set data value locations for aiming test */
  {
    dx=MaxX/3;
    Mines_X=x=MaxX/6-2*textwidth( "H" );
    x=x+dx;   Speed_X=x;
    x=x+dx-textwidth("H"); Score_X=x;
  }
	/* set graphics eraser is done in main */
}

void Close_Graphics(void)
{
  cleardevice();
  restorecrtmode();
}

float Fcos(int Headings_Degs) /* compute cos of 0 - 359 degrees */
  {
    float arc;
    arc=Headings_Degs*ARC_CONV;
    return(cos(arc));
  }

float Fsin(int Headings_Degs) /* compute sin of 0 - 359 degrees */
  {
    float arc;
    arc=Headings_Degs*ARC_CONV; /* convert degrees to radians */
    return(sin(arc));
  }

void Draw_Frame()
{
  int Height;
  int t,t1,svcolor;
  int x,y,dx;

  Height=textheight("H");		/* Get basic text height */

  svcolor=getcolor();                   /* save present color */
  setbkcolor(BACKGROUND_COLOR);
  cleardevice();
  setcolor(FRAME_COLOR);
	/* handle panel */
  setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);
	/* data panel in screen global coordinates */
  rectangle(0,0,MaxX,MaxY_Panel);
  line(0,2*Height,MaxX,2*Height);

		  /* write panel headers */
  if(Game_Type==SPACE_FORTRESS)
    {
      x=2;
      y=4;
      dx=MaxX/8; /* step between two headers */
      Data_Line=2*Height+4;
      gprintf ( &x, &y,"  PNTS");
      x=x+dx; gprintf ( &x, &y," CNTRL");
      x=x+dx; gprintf ( &x, &y," VLCTY");
      x=x+dx; gprintf ( &x, &y," VLNER");
      x=x+dx; gprintf ( &x, &y,"  IFF ");
      x=x+dx; gprintf ( &x, &y,"INTRVL");
      x=x+dx; gprintf ( &x, &y," SPEED");
      x=x+dx; gprintf ( &x, &y," SHOTS");

	  /* draw vertical lines between columns */

	      line(x,0,x,MaxY_Panel);
      x=x-dx; line(x,0,x,MaxY_Panel);
      x=x-dx; line(x,0,x,MaxY_Panel);
      x=x-dx; line(x,0,x,MaxY_Panel);
      x=x-dx; line(x,0,x,MaxY_Panel);
      x=x-dx; line(x,0,x,MaxY_Panel);
      x=x-dx; line(x,0,x,MaxY_Panel);
   }
   else /* frame for aiming test */
   {
     x=MaxX/6-32;
     y=4;
     dx=MaxX/3;		 /* step between two headers */
     Data_Line=2*Height+4;
	  gprintf ( &x, &y,"  MINES");
     x=x+dx; gprintf ( &x, &y," SPEED");
     x=x+dx; gprintf ( &x, &y," SCORE");

	  /* draw vertical lines between columns */
     x=dx;   line(x,0,x,MaxY_Panel);
     x=x+dx; line(x,0,x,MaxY_Panel);
   }

  setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1); /* in screen global coordinates */
  rectangle(0,0,MaxX,MaxY); 		/* main frame of the viewport */

  /* set graphics eraser is done in main */

  setcolor(svcolor); /* restore previous color */
}

int Draw_Hexagone(int X_Center,int Y_Center,int Hex_Outer_Radius)
{
  int Abs_Y;
  int svcolor;

  svcolor=getcolor(); /* save present color */
  setcolor(HEX_COLOR);

  Abs_Y=Hex_Outer_Radius*0.866;  /* sin(60)=0.866 */
  moveto(X_Center+Hex_Outer_Radius,Y_Center); /* right-hand tip */
  lineto(X_Center+Hex_Outer_Radius/2,Y_Center-Abs_Y);
  lineto(X_Center-Hex_Outer_Radius/2,Y_Center-Abs_Y);

  lineto(X_Center-Hex_Outer_Radius,Y_Center);
  lineto(X_Center-Hex_Outer_Radius/2,Y_Center+Abs_Y);

  lineto(X_Center+Hex_Outer_Radius/2,Y_Center+Abs_Y);
  lineto(X_Center+Hex_Outer_Radius,Y_Center);

  setcolor(svcolor); /* restore previous color */
  return(0);
}

int  Draw_Ship (int x, int y, int Headings, int size)
{
  /* size - is the entire length of the ship */
  int x1,y1;  /* ship's aft location */
  int x2,y2;  /* ship's nose location */
  int xl,yl;     /* ship's left wing tip location */
  int xr,yr;     /* ship's right wing tip location */
  int xc,yc;  /* fuselage and wings connecting point */
  int Right_Wing_Headings;
  int Left_Wing_Headings;
  int svcolor;
  float tmp;

  svcolor=getcolor(); /* save present color */
  setcolor(SHIP_COLOR);
  xc=x;
  yc=y;
  x1=xc-0.5*size*Fsin(Headings);
  y1=yc+0.5*size*Fcos(Headings);
  x2=xc+0.5*size*Fsin(Headings);
  y2=yc-0.5*size*Fcos(Headings);
  line(x1,y1,x2,y2);
  Right_Wing_Headings=Headings+135;
  if(Right_Wing_Headings>359) Right_Wing_Headings=Right_Wing_Headings-360;
  Left_Wing_Headings=Headings+225;
  if(Left_Wing_Headings>359) Left_Wing_Headings=Left_Wing_Headings-360;
  xr=xc+0.707*size*Fsin(Right_Wing_Headings);
  yr=yc-0.707*size*Fcos(Right_Wing_Headings);
  line(xc,yc,xr,yr);
  xl=xc+0.707*size*Fsin(Left_Wing_Headings);
  yl=yc-0.707*size*Fcos(Left_Wing_Headings);
  line(xc,yc,xl,yl);
  setcolor(svcolor); /* restore previous color */
  return(0);
}

int Draw_Fort (int x, int y, int Headings, int size )
{
  int x1,y1;     /* fort's aft location */
  int x2,y2;     /* fort's nose location */
  int xl,yl;     /* ship's left wing tip location */
  int xr,yr;     /* ship's right wing tip location */
  int xc,yc;     /* fuselage and wings connecting point */
  int xrt,yrt;   /* tip of right wing */
  int xlt,ylt;   /* tip of left wing */
  int Right_Wing_Headings;
  int Left_Wing_Headings;
  int Right_Wing_Tip_Headings;
  int Left_Wing_Tip_Headings;
  int svcolor;

  svcolor=getcolor(); /* save present color */
  setcolor(FORT_COLOR);
  x1=x;
  y1=y;
  x2=x1+size*Fsin(Headings);
  y2=y1-size*Fcos(Headings);
  line(x1,y1,x2,y2);
  xc=x1+(x2-x1)*0.5;
  yc=y1+(y2-y1)*0.5;
  Right_Wing_Headings=Headings+90;
  if(Right_Wing_Headings>359) Right_Wing_Headings=Right_Wing_Headings-360;
  Left_Wing_Headings=Headings+270;
  if(Left_Wing_Headings>359) Left_Wing_Headings=Left_Wing_Headings-360;
  xr=xc+0.4*size*Fsin(Right_Wing_Headings)+0.5;
  yr=yc-0.4*size*Fcos(Right_Wing_Headings)+0.5;
  line(xc,yc,xr,yr);
  xl=xc+0.4*size*Fsin(Left_Wing_Headings)+0.5;
  yl=yc-0.4*size*Fcos(Left_Wing_Headings)+0.5;
  line(xc,yc,xl,yl);
  Right_Wing_Tip_Headings=Right_Wing_Headings+90;
  if(Right_Wing_Tip_Headings>359) Right_Wing_Tip_Headings=
				       Right_Wing_Tip_Headings-360;
  xrt=xr+0.5*size*Fsin(Right_Wing_Tip_Headings)+0.5;
  yrt=yr-0.5*size*Fcos(Right_Wing_Tip_Headings)+0.5;
  line(xr,yr,xrt,yrt);
  Left_Wing_Tip_Headings=Right_Wing_Tip_Headings;
  xlt=xl+0.5*size*Fsin(Left_Wing_Tip_Headings)+0.5;
  ylt=yl-0.5*size*Fcos(Left_Wing_Tip_Headings)+0.5;
  line(xl,yl,xlt,ylt);
  setcolor(svcolor); /* restore previous color */
  return(0);
}

int Draw_Mine (int x, int y, int size)  /* x,y is on screen center location
					size is half of horizontal diagonal */
{
  int svcolor;

  svcolor=getcolor(); /* save present color */
  setcolor(MINE_COLOR);

  moveto(x-size,y);
  lineto(x,y-1.18*size);   /* 1.3/1.1=1.18 */
  lineto(x+size,y);
  lineto(x,y+1.18*size);
  lineto(x-size,y);
  setcolor(svcolor); /* restore previous color */
  return(0);
}

int Draw_Missile (int x, int y, int Headings, int size)
{
  int x1,y1;  /* ship's aft location */
  int x2,y2;  /* ship's nose location */
  int xl,yl;     /* ship's left wing tip location */
  int xr,yr;     /* ship's right wing tip location */
  int xc,yc;  /* fuselage and wings connecting point */
  int Right_Wing_Headings;
  int Left_Wing_Headings;
  int svcolor;

  svcolor=getcolor(); /* save present color */
  setcolor(MISSILE_COLOR);
  x1=x;
  y1=y;
  x2=x1+size*Fsin(Headings);
  y2=y1-size*Fcos(Headings);
  line(x1,y1,x2,y2);
  xc=x2;
  yc=y2;
  Right_Wing_Headings=Headings+135;
  if(Right_Wing_Headings>359) Right_Wing_Headings=Right_Wing_Headings-360;
  Left_Wing_Headings=Headings+225;
  if(Left_Wing_Headings>359) Left_Wing_Headings=Left_Wing_Headings-360;
  xr=xc+0.25*size*Fsin(Right_Wing_Headings);
  yr=yc-0.25*size*Fcos(Right_Wing_Headings);
  line(xc,yc,xr,yr);
  xl=xc+0.25*size*Fsin(Left_Wing_Headings);
  yl=yc-0.25*size*Fcos(Left_Wing_Headings);
  line(xc,yc,xl,yl);
  setcolor(svcolor); /* restore previous color */
  return(0);
}

int Draw_Shell(int x, int y, int Headings, int size)
{
  int x1,y1;  /* shell's aft location */
  int x2,y2;  /* shell's nose location */
  int xl,yl;     /* shell's left tip location */
  int xr,yr;     /* shell's right tip location */
  int Right_Tip_Headings;
  int Left_Tip_Headings;
  int svcolor;

  svcolor=getcolor(); /* save present color */
  setcolor(SHELL_COLOR);
  x1=x;
  y1=y;
  x2=x1+size*Fsin(Headings);
  y2=y1-size*Fcos(Headings);
  Right_Tip_Headings=Headings+30;
  if(Right_Tip_Headings>359) Right_Tip_Headings=Right_Tip_Headings-360;
  Left_Tip_Headings=Headings+330;
  if(Left_Tip_Headings>359) Left_Tip_Headings=Left_Tip_Headings-360;
  xr=x1+0.4*size*Fsin(Right_Tip_Headings);
  yr=y1-0.4*size*Fcos(Right_Tip_Headings);
  xl=x1+0.4*size*Fsin(Left_Tip_Headings);
  yl=y1-0.4*size*Fcos(Left_Tip_Headings);
  moveto(x1,y1);
  lineto(xl,yl);
  lineto(x2,y2);
  lineto(xr,yr);
  lineto(x1,y1);
  setcolor(svcolor); /* restore previous color */
  return(0);

}

int Find_Headings(int x1,int y1,int x2,int y2)
{
  int quadrant;
  double arcsinalfa;
  double b;
  double a;
  arcsinalfa=abs(x1-x2);
  a=pow(x1-x2,2)+pow(y1-y2,2);
  b=sqrt(a);
  arcsinalfa=asin(arcsinalfa/b);
  if (x1<x2)
     if (y1>y2) /* quadrant=1 */ return(arcsinalfa*57.3+0.5);
	else
		/* quadrant=2 */ return(180-arcsinalfa*57.3+0.5);
     else
     if (y1>y2) /* quadrant=4 */ return(360-arcsinalfa*57.3+0.5);
	else
		/* quadrant=3 */ return(180+arcsinalfa*57.3+0.5);

}

