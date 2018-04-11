
/* test graphics 21.2.90 18:00
            definitions */
#include <stdarg.h>
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
#include "myvars.h"

extern char Friend_Menu[3][1];
extern char Foe_Menu[3][1];
extern char Mine_Indicator;
extern mine_type Mine_Type;

        /* Functions  */
extern void Open_Graphics(void);
extern void Close_Graphics(void);
extern float Fcos(int Headings_Degs);
extern float Fsin(int Headings_Degs);
extern Draw_Ship (int x, int y, int Headings, int size);
extern int Draw_Hexagone(int X_Center,int Y_Center,int Hex_Size);
extern void Draw_Frame();
extern Draw_Fort (int x, int y, int Headings, int size );
extern Draw_Missile (int x, int y, int Headings, int size);
extern Draw_Shell(int x, int y, int Headings, int size);
extern Draw_Mine (int x, int y, int size);
extern int Find_Headings(int x1,int y1,int x2,int y2);
extern Update_Ship_Dynamics();
extern Update_Ship_Display();
extern Move_Ship();
extern Fire_Shell();
extern Handle_Fortress();
extern Test_Collisions();
extern Generate_Mine();
extern Move_Mine();
extern Handle_Mine();
extern Handle_Shell();
extern Handle_Missile();
extern Accumulate_Data();
extern Push_Buttons();
extern Joystick();
extern Setup_Mouse_Handler();
extern Reset_Mouse_Handler();
extern Initialize_Graphics();


void Mydelay(unsigned Time)
{
    unsigned long end;

    end=Time_Counter+Time;
    while(Time_Counter<end);

}

void interrupt far Get_Tik()
{
    Time_Counter++;
    oldtik(); /*  now perform the old BIOS handler to keep things clean */
}

    /****** capture system clock tiks via indicated routine **********/

void Capture_Tik(void interrupt far (*func) () )
{
    /* save old interrupt */
    oldtik=getvect(8);
    /* install our new interrupt handler */
    disable();
    setvect(8,func);
    enable();
}

Restore_Tik()
{
    disable();
    setvect(8,oldtik);   /* restore old interrupt handler */
    enable();
    return(0);
}

Set_Timer()
{
    outportb(0x43,0x36);
    outportb(0x40,1193&0xFF); /* 100 Hz */
    outportb(0x40,1193>>8);
    return(0);
}

Reset_Timer()
{
    outportb(0x43,0x36);  /* 36 talk to control register */
    outportb(0x40,0xFFFF);
    outportb(0x40,0xFFFF>>8);
    return(0);
}

Set_Kbd_Rate(unsigned char Rate)
{
    _AH=0x3;
    _AL=0x5;
    _BH=0;
    _BL=Rate;   /* repeat rate of 20 Hz */
    geninterrupt(0x16);
    return(0);
}

int keyboard (void)
{
    union u_type{int a; char b[3];} keystroke;
    char inkey=0;
    char xtnd=0;

    if(bioskey(1)==0) return(NO_INPUT);  /* key relieved, no input */
    keystroke.a=bioskey(0);   /* fetch ascii code */
    inkey=keystroke.b[0];     /* ..and load code into variable */

    if(inkey==27) return(ESC); /* ESCcape terminates program */
    if(inkey==13) return(ENTER); /* ENTER pauses program */
    if(inkey==97) return(F1);
    if(inkey==115) return(F2);
    if(inkey==100) return(F3);
    if(inkey==32) return(DOWN);
    if(inkey==8) Restart_Flag=ON;
    if(inkey!=0) return(REGULAR_CRAP);   /* the rest is crap */
    
    /* which leaves inkey==0 */
    xtnd=keystroke.b[1];
        
        switch (xtnd)
         {
             case 72 /*UP*/    : return(UP);
             case 75 /*LEFT*/  : return(LEFT);
             case 77 /*RIGHT*/ : return(RIGHT);
             default           : return(EXTENDED_CRAP);/* all rest irrelevant */
    }
}

void interrupt far Get_Key() {
    int tmp;
    oldfunc(); /* now perform the old BIOS handler to keep things clean */

    tmp=keyboard();
    if(tmp) {        /* throw away zeros, they indicate key release! */
        New_Input_Flag=ON;
        Lastkey=Key;
        Key=tmp;

        /***** now handle double press time interval measurement  ******/

        if(Key==F3) {
            if((Key==F3)&&(Lastkey!=F3)&&(!(Timing_Flag))) { /* first F3 keypress */
                t1=Time_Counter;
                Timing_Flag=ON;
                Check_Mine_Flag=ON; /* is used by Get_User_Input() */
            }

            if((Key==F3)&&(Lastkey==F3)&&(Timing_Flag)) {   /* second F3 keypress */
                t2=Time_Counter;
                Timing_Flag=OFF;
                Key=0;   /* to enable consecutive double_press */
                /* where with next keypress Lastkey=0 */
                Display_Interval_Flag=ON;  /* is used in main */
            }
            New_Input_Flag=OFF;   /* input was handled here */
        } /* end double press */
    } /* end if(tmp) */
}

/****** capture any keyboard input via indicated routine **********/
void Capture_Kbd(void interrupt far (*func) () )
{
    /* save old interrupt */
    oldfunc=getvect(9);
    /* install our new interrupt handler */
    disable();
    setvect(9,func);
    enable();
}

Restore_Kbd()
{
    disable();
    setvect(9,oldfunc);   /* restore old interrupt handler */
    enable();
    return(0);
}

void Get_User_Input()
{
    if (New_Input_Flag) /* new input occured */
    {
        New_Input_Flag=OFF; /* make sure no repetitions on same input */
        if (Key==UP)    Accel_Input=1;        /*   UP    */
        if (Key==LEFT)  Rotate_Input=-1;      /*   LEFT  */
        if (Key==RIGHT) Rotate_Input=1;       /*   RIGHT */
        if (Key==DOWN)  New_Missile_Flag=ON;  /*   DOWN  */
        if (Key==F1)    Check_Bonus_Input();        /*   P(oints) */
        if (Key==F2)    Check_Bonus_Input();        /*   M(issiles) */
/*    if (Key==F3)    is handled by kbd interrupt handler */
        if (Key==ENTER) Freeze_Flag=Freeze_Flag^1; /* toggle freeze flag */
        if (Key==ESC)   End_Flag=ON;
    }
    if(Check_Mine_Flag) /* after first press of F3 */
        {
            Check_Mine_Flag=OFF;
            if((Mine_Flag==ALIVE) && (Mine_Type==FRIEND))
    Missile_Type=WASTED;
        Show_Mine_Type(Mine_Indicator);
        }
}


char Keyboard1() /* handles escape key press only */
{
    union u_type{int a; char b[3];} keystroke;
    char inkey=0;

    while(bioskey(1)==0);   /* key relieved, no input */
    keystroke.a=bioskey(0); /* fetch ascii code */
    inkey=keystroke.b[0];   /* ..and load code into variable */
    return(inkey);

}

int gprintf( int *xloc, int *yloc, char *fmt, ... )
{
    va_list  argptr;      /* Argument list pointer  */
    char str[140];      /* Buffer to build sting into */
    int cnt;        /* Result of SPRINTF for return */

    va_start( argptr, fmt );    /* Initialize va_ functions */
    cnt = vsprintf( str, fmt, argptr ); /* prints string to buffer  */
    outtextxy( *xloc, *yloc, str ); /* Send string in graphics mode */
    va_end( argptr );     /* Close va_ functions    */
    return( cnt );      /* Return the conversion count  */
}

Set_Graphics_Eraser()
{
    int size;

    size=imagesize(0,0,40,9);        /*length of 5 characters*/
    buffer1=malloc(size);
    getimage(100,100,140,109,buffer1);
    return(0);
}

Show_Score(int val, int x, int y) /* anywhere within data panel */
{
    int svcolor;

    svcolor=getcolor();
    setcolor(TEXT_COLOR);
    setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);

    /* data panel in screen global coordinates */

    putimage(x,y,buffer1,COPY_PUT); /* erase garbage */
    gprintf(&x,&y,"%d",val);

    setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);   /* restore gaming area */
    setcolor(svcolor); /* restore previous color */
    return(0);
}

Update_Points()
{
    Show_Score(Points,Points_X-8,Data_Line);
    return(0);
}

Update_Control()
{
    Show_Score(Control,Control_X-8,Data_Line);
    return(0);
}

Update_Velocity()
{
    Show_Score(Velocity,Velocity_X,Data_Line);
    return(0);
}

Update_Vulner()  /* for vulner only */
{
    Show_Score(Vulner_Counter,Vulner_X,Data_Line);
    return(0);
}

/* IFF is missing here */

Update_Interval()
{
    Show_Score(Double_Press_Interval,Interval_X,Data_Line);
    return(0);
}

Update_Speed()
{
    Show_Score(Speed,Speed_X-8,Data_Line);
    return(0);
}

Update_Shots()
{
    Show_Score(Missile_Stock,Shots_X,Data_Line);
    return(0);
}

Clear_Interval()   /* clear double-press interval */
{
        int svcolor;
        int x,y;

        svcolor=getcolor();
        setcolor(TEXT_COLOR);
        setviewport( Xmargin, Panel_Y_Start, Xmargin+MaxX, Panel_Y_End, 1);
        x=Interval_X; y=Data_Line;
        putimage(x,y,buffer1,COPY_PUT); /* erase garbage */
        setviewport( Xmargin, 0, Xmargin+MaxX, MaxY, 1);   /* restore gaming area */
        setcolor(svcolor); /* restore previous color */
        return(0);
}

Find_Interval()   /* display double-press interval */
{
    int svcolor;
    int x,y;
    int interval;

    interval=Double_Press_Interval=t2-t1; /* in milliseconds */
    if((interval<DELAY*20)&&(interval>DELAY)) /* only when interval makes sense */
    {
        if((interval>=Interval_Lower_Limit)&&(interval<=Interval_Upper_Limit)
             &&(Mine_Flag==ALIVE)&&(Mine_Type==FOE))
    Missile_Type=VS_FOE;   /* rearm missile */
        Show_Mine_Type(Mine_Indicator);
        Update_Interval();
    }
    return(0);
}

void Reset_Screen()
{
    int i;
        /*  reset variables */
    Ship_X_Pos=0.25*MaxX; /* on a 640 x 480 screen VGA-HI */
    Ship_Y_Pos=0.5*MaxY; /* same as above */
    Ship_X_Speed=0.0;
    Ship_Y_Speed=0.0;
    Ship_Headings=0;
    Mine_Flag=DEAD;
    for(i=1;i<6;i++) Missile_Flag[i]=DEAD;
    Missile_Type=VS_FRIEND;
    Missile_Vs_Mine_Only=OFF;
    Missiles_Counter=0;
    Shell_Flag=DEAD;
    Rotate_Input=0; /* joystick left/right */
    Accel_Input=0; /* joystick forward */
    End_Flag=OFF;
    Fort_Headings=270;
    Vulner_Counter=0;
    Timing_Flag=OFF; /* if screen reset between consecutive presses */
    Resource_Flag=OFF;
    Resource_Off_Counter=0;
    Bonus_Display_Flag=NOT_PRESENT;   /* in case bonus is pressed after
    Bonus_Granted=OFF;
    Fort_Lock_Counter=0;

        /* reset screen */
    Draw_Frame();
    if(AspectRatio==1.0)
    {
        Draw_Hexagone(MaxX/2,MaxY/2,BIG_HEXAGONE_SIZE_FACTOR*MaxX);
        Draw_Hexagone(MaxX/2,MaxY/2,SMALL_HEXAGONE_SIZE_FACTOR*MaxX);
    }
    else
    {
        Draw_Hexagone(MaxX/2,MaxY/2,BIG_HEXAGONE_SIZE_FACTOR*MaxX/GraphSqrFact);
        Draw_Hexagone(MaxX/2,MaxY/2,SMALL_HEXAGONE_SIZE_FACTOR*MaxX/GraphSqrFact);
    }
    Draw_Fort(MaxX/2,MaxY/2,Fort_Headings,FORT_SIZE_FACTOR*MaxX);
    Draw_Ship(Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);

            /* reset panel */
    Update_Points();
    Update_Vulner();
    Update_Interval();
    Update_Shots();
    Update_Control();
    Update_Velocity();
    Update_Speed();

}  /* end reset screen */

Init_Session() {
    One_Game_Loops=One_Game_Duration*60*20;
    Game_Type=SPACE_FORTRESS;
    Mine_Live_Loops=200;
    Mine_Wait_Loops=80;
    return(0);
}

Init_Game()
{
    int i;

    Score=0;
    Points=0;
    Velocity=0;
    Control=0;
    Speed=0;
    No_Of_Bonus_Intervals=6;
    No_Of_Points_Bonus_Taken=0;
    No_Of_Missiles_Bonus_Taken=0;
    Ship_Damaged_By_Fortress=0;
    Ship_Damaged_By_Mines=0;
    Fortress_Destroyed=0;
    Normal_Game_Termination=0;
    Vulner_Counter=0;
    Last_Missile_Hit=0; /* to measure interval between two consecutive
                    hits of the fortress */
    Ship_Killings_Counter=0;
    Resource_Flag=OFF;
    Resource_Off_Counter=0;
    Bonus_Display_Flag=NOT_PRESENT;   /* in case bonus is pressed
                    after game ends */
    No_Of_Bonus_Windows=0;
    Missile_Stock=100;

    Select_Mine_Menus();
    /*
    clrscr();
    gotoxy(30,5);
    printf("SPACE  FORTRESS ");
    gotoxy(20,15);
    Select_Mine_Menus();
    printf("Your foe mines are:");
    for(i=0;i<3;i++) printf("    %c",Foe_Menu[i][0]);
    gotoxy(1,24);
    printf("Type any  key to continue ..\n");
    getch();*/
    return(0);
}

Display_Bonus_Char(char Bonus_Char)
{
    int svcolor;
    int x,y;
    svcolor=getcolor();
    setcolor(TEXT_COLOR);
    settextstyle(DEFAULT_FONT,HORIZ_DIR,2);
    x=MaxX/2 - 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
    y=MaxY/2 + 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
    gprintf(&x,&y,"%c",Bonus_Char);
    settextstyle(DEFAULT_FONT,HORIZ_DIR,0);
    setcolor(svcolor); /* restore previous color */
    return(0);
}

Set_Bonus_Chars()
{
    int size,i,j;
    int x,y;

    /* set character Size */
    size=imagesize(0,0,16,16);
    /* get right location */
    x=MaxX/2 - 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
    y=MaxY/2 + 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;

 for (i=0;i<10;i++)
         {
        bc[i]=malloc(size);
        Display_Bonus_Char(Bonus_Char_Vector[i][0]);
        getimage(x,y,x+16,y+16,bc[i]);
        putimage(x,y,bc[i],XOR_PUT);
         }
 return(0);
}

Xor_Bonus_Char(int n)   /* write and erase bonus character */
{
    int x,y;

    /* get right location */
    x=MaxX/2 - 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
    y=MaxY/2 + 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;

    putimage(x,y,bc[n],XOR_PUT);
    return(0);
}

Set_Bonus_Message()
{
    int size;
    int svcolor;
    int x,y;

    svcolor=getcolor();
    setcolor(TEXT_COLOR);
    x=MaxX/2 - 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
    y=MaxY/2 + 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
    gprintf(&x,&y,"Bonus");
    setcolor(svcolor); /* restore previous color */

    size=imagesize(0,0,40,9);        /*length of 5 characters*/
    buffer2=malloc(size);
    getimage(x,y,x+40,y+9,buffer2);
    putimage(x,y,buffer2,XOR_PUT);
    setcolor(svcolor);
    return(0);
}

Write_Bonus_Message()
{
 int x,y;

 x=MaxX/2 - 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
 y=MaxY/2 + 1.2*SMALL_HEXAGONE_SIZE_FACTOR*MaxX;
 putimage(x,y,buffer2,XOR_PUT);
 return(0);
}

Check_Bonus_Input() {
    if((Bonus_Display_Flag==NOT_PRESENT)||(Bonus_Display_Flag==NON_BONUS)) {
    } else if(Bonus_Display_Flag==FIRST_BONUS) {
        Bonus_Wasted_Flag=ON;
    } else if(Bonus_Display_Flag==SECOND_BONUS) {
        if(!Bonus_Wasted_Flag) {
            if(Key==F1) {
                No_Of_Points_Bonus_Taken++;
                Points=Points+100;
                Update_Points();
            } else {
                No_Of_Missiles_Bonus_Taken++;
                Missile_Stock=Missile_Stock+50;
                if(Missile_Stock>=100) Missile_Stock=100;
                Update_Shots();
            }
        Bonus_Display_Flag=NOT_PRESENT;
        Bonus_Granted=ON;
        Xor_Bonus_Char(rn);    /* erase present $ char */
        Write_Bonus_Message(); /*  Announce_Bonus  */
        }
    }
return(0);
}

int Generate_Non_Bonus_Char()
{
     int rn;

     do { rn=random(10); }
     while(rn==Bonus_Indication_Index);
     return(rn);
}

Generate_Resource_Character()
{
    int lastrn;
    static bonus_character lastchar=NON_BONUS;

    if((lastchar==NON_BONUS)&&
         (No_Of_Bonus_Windows<MAX_BONUS_WINDOWS))

             if(random(10)<7) /* display first bonus */
     {
         No_Of_Bonus_Windows++;
         rn=Bonus_Indication_Index;
         Xor_Bonus_Char(rn);
         lastchar=Bonus_Display_Flag=FIRST_BONUS;
         Bonus_Wasted_Flag=OFF;
     }
             else /* display non_bonus character */
     {
         lastrn=rn;
         do { rn=Generate_Non_Bonus_Char(); }
         while(rn==lastrn); /* new char is different from last one */
         Xor_Bonus_Char(rn);
         lastchar=Bonus_Display_Flag=NON_BONUS;
     }
    else
    if(lastchar==FIRST_BONUS)
        {
            Xor_Bonus_Char(rn);
            lastchar=Bonus_Display_Flag=SECOND_BONUS;
        }
    else
    if(lastchar==SECOND_BONUS)
        {
             rn=Generate_Non_Bonus_Char();
             Xor_Bonus_Char(rn);
             lastchar=Bonus_Display_Flag=NON_BONUS;
        }
return(0);
}

Handle_Bonus()
{

    if(!Resource_Flag)   /* resource is off */
        {
            Resource_Off_Counter++;
            if(Resource_Off_Counter>=No_Resource_Display_Interval)
    {
        Resource_Flag=ON;
        Resource_On_Counter=0;
        Generate_Resource_Character();
    }
     }
 else   /* Resource_Flag=ON; */
     {
         Resource_On_Counter++;
         if(Resource_On_Counter>=Resource_Display_Interval)
             {
     Resource_Flag=OFF;
     Resource_Off_Counter=0;
     Bonus_Display_Flag=NOT_PRESENT; /* in case bonus is pressed after
                            $ disappears */
     if (Bonus_Granted)
            {
                Write_Bonus_Message();     /* erase bonus message */
                Bonus_Granted=OFF;
            }
     else
     Xor_Bonus_Char(rn);  /* Erase_Resource_Char */
             }
     }
return(0);
}

Run_SF()
{
    unsigned elapsed_time;
    unsigned long loop_start_time;

    // SCORE SAVE FILE
    FILE *f = fopen("SAVE\\SCORE.TXT", "w");
    if (f == NULL) {
        printf("A state file is not present.\n");
        exit(1);
    }
    fclose(f);

    Init_Session();
    Game_Counter=0;
    do {   /* loop on number of games here */
        Init_Game();
        Open_Graphics();
        Initialize_Graphics();
        Reset_Screen();
        
        Loop_Counter=0;
        Set_Kbd_Rate(0x8); /* to slow repeat rate 15Hz */
        Capture_Kbd(Get_Key); /* redirect KBD interrupts to  Get_Key() */
        Time_Counter=0;
        Capture_Tik(Get_Tik);
        Set_Timer();

        do {   /* real time loop of one game */
            loop_start_time=Time_Counter;
            Loop_Counter++;
            Get_User_Input();
            while(Freeze_Flag) Get_User_Input();
            Move_Ship();
            Handle_Missile();
            if(Sound_Flag>1) Sound_Flag--;
            if(Sound_Flag==1) {Sound_Flag--; nosound();}
            Handle_Mine();
            Test_Collisions();
            Handle_Shell();
            Handle_Fortress();
            if(Display_Interval_Flag) {   /* of double press */
                if(Mine_Type==FOE) Find_Interval();
                Display_Interval_Flag=OFF;
            }
            Accumulate_Data();
            Handle_Bonus();
            if(!Effect_Flag) {
                if((elapsed_time=Time_Counter-loop_start_time) < DELAY)
                    Mydelay(DELAY-elapsed_time);  /* wait up to 50 milliseconds */
            } else Effect_Flag=OFF;  /* no delay necessary */

            Score=Points+Velocity+Control+Speed;

            // SAVE SCORE
            f = fopen("SAVE\\SCORE.TXT", "w");
            fprintf(f, "%d", Score);
            fclose(f);

        } while(!Restart_Flag&&!End_Flag&&(Loop_Counter < One_Game_Loops));
        /* ESC or three minutes */

        // RUNNING FILE
        f = fopen("SAVE\\SCORE.TXT", "w");
        fprintf(f, "FALSE");
        fclose(f);

        Restore_Tik();
        Reset_Timer();
        Restore_Kbd();
        Set_Kbd_Rate(0x4); /* to repeat rate 20Hz */

        nosound();   /* just in case */
        sound(400);
        delay(500);
        nosound();
        Game_Counter++;
        
        // final_display();
        Close_Graphics();
        printf("Episode %d score: %d\n", Game_Counter, Score);
        if(!Restart_Flag && !End_Flag) {
            while(1) {
                char ex = getch();
                if(ex==9) {
                    break;
                } else if(ex==27) {
                    return(0);
                }
            }
        }

        
        clrscr();
        /* end one game here */
    } while(!Restart_Flag && !End_Flag);
    //} while((Game_Counter< No_Of_Games)&&(!End_Flag));

    nosound();   /* just in case */
    sound(400);
    delay(1000);
    nosound();
    if(Restart_Flag) {
        return(1);
    }
    return(0);
}

/*************************************************************************/

        /* Run_Aiming Module  4.4.90 */

/**************************************************************************/

Announce_Game_End()
{
    int svcolor;
    int x,y;

    svcolor=getcolor();
    setcolor(TEXT_COLOR);
    x=0.35*MaxX; y=0.7*MaxY;
    gprintf(&x,&y,"GAME IS OVER !");
    x=0.25*MaxX; y=0.8*MaxY;
    gprintf(&x,&y,"<press any key to continue>");
    setcolor(svcolor); /* restore previous color */
    return(0);
}

Announce_Session_End()
{
    int svcolor;
    int x,y;

    svcolor=getcolor();
    setcolor(TEXT_COLOR);
    x=0.35*MaxX; y=0.7*MaxY;
    gprintf(&x,&y,"SESSION IS OVER !");
    x=0.25*MaxX; y=0.8*MaxY;
    gprintf(&x,&y,"<press any key to continue>");
    setcolor(svcolor); /* restore previous color */
    return(0);
}

Update_Mines()
{
    Show_Score(Mines,Mines_X,Data_Line);
    return(0);
}

Update_Score()
{
    Show_Score(Score,Score_X,Data_Line);
    return(0);
}

void Reset_Aim_Screen()
{
    int i;
        /*  reset variables */
    Ship_X_Pos=0.5*MaxX; /* on a 640 x 480 screen VGA-HI */
    Ship_Y_Pos=0.5*MaxY; /* same as above */
    Ship_X_Speed=0.0;
    Ship_Y_Speed=0.0;
    Ship_Headings=0;
    Mine_Flag=DEAD;
    for(i=1;i<6;i++) Missile_Flag[i]=DEAD;
    Missile_Type=VS_FRIEND;
    Missile_Vs_Mine_Only=OFF;
    Missiles_Counter=0;
    Shell_Flag=DEAD;
    Rotate_Input=0; /* joystick left/right */
    Accel_Input=0; /* joystick forward */
    End_Flag=OFF;
    Restart_Flag=OFF;
    Fort_Headings=270;
    Timing_Flag=OFF; /* if screen reset between consecutive presses */
        /* reset screen */
    Draw_Frame();
    Draw_Ship(Ship_X_Pos,Ship_Y_Pos,Ship_Headings,SHIP_SIZE_FACTOR*MaxX);

            /* reset panel */
    Update_Mines();
    Update_Speed();
    Update_Score();
}  /* end reset screen */

Init_Aim_Session()
{
    char a;
    int i;

    clrscr();
    gotoxy(30,5);
    printf("AIMING TASK TEST");
    gotoxy(1,8);
    printf("The default # of games is 3, would you like to change it? (Y/N):");
    a=Keyboard1();
    if((a==121)||(a==89))  /* y=121 and Y=89 */
        {
         printf("\n\nPlease type new number of games (1-9):");
         do { a=Keyboard1();
        if((a>=49)&&(a<=57)) { No_Of_Games=a-48;
                 printf(" %d\n",No_Of_Games);
                 delay(500);
                         }
        else a=0;
    } while(a==0);
        }
    clrscr();

    Game_Type=AIMING_TEST;
    Mine_Live_Loops=200;
    Mine_Wait_Loops=10;
    return(0);
}

Save_Aiming_Game() {
    header.Number_Of_Planned_Games=No_Of_Games;
    header.One_Game_Duration=One_Game_Duration;

    Aiming_Game_Results.mines=Mines;
    Aiming_Game_Results.speed=Speed;
    Aiming_Game_Results.score=Score;

    return(0);
}

Run_Aiming()   /* 1- for training 0- for demo */
{
    unsigned elapsed_time;
    unsigned long loop_start_time;

    Init_Aim_Session();
    Open_Graphics();
    Initialize_Graphics();
    Set_Graphics_Eraser();
    Game_Counter=0; /* no of games played */
    Set_Kbd_Rate(0x8); /* to slow repeat rate .. */
    do
    {
    Game_Counter++;
    Mines=0; Speed=0; Score=0;
    Reset_Aim_Screen();
    Loop_Counter=0;
    Capture_Kbd(Get_Key); /* redirect KBD interrupts to  Get_Key() */
    Time_Counter=0;
    Capture_Tik(Get_Tik);
    Set_Timer();
    do
        {
         loop_start_time=Time_Counter;
         Loop_Counter++;
         Get_User_Input();
            while(Freeze_Flag) Get_User_Input();
         Move_Ship();  /* rotation only */
         Handle_Missile();
         if(Sound_Flag>1) Sound_Flag--;
         if(Sound_Flag==1) {Sound_Flag--; nosound();}
         Handle_Aim_Mine();
         Test_Collisions();
         if(!Effect_Flag)
             {
     if ( (elapsed_time=Time_Counter-loop_start_time) < DELAY)
            Mydelay(DELAY-elapsed_time);  /* wait up to 50 milliseconds */
             }
         else Effect_Flag=OFF;  /* no delay necessary */
        } while((!End_Flag)&&(Loop_Counter < 2400));
    /* ESC or three minutes */

    Restore_Tik();
    Reset_Timer();
    Restore_Kbd();
    Set_Kbd_Rate(0x4); /* to repeat rate 20Hz */
    if((!End_Flag)&&(Game_Counter<No_Of_Games))
                     { Announce_Game_End();
                         nosound();   /* just in case */
                         sound(600);
                         delay(1000);
                         nosound();
                         while(keyboard()); /* clear keyboard */
                         getch();
                     }
    Set_Kbd_Rate(0x4); /* to slow repeat rate 15Hz */
    Save_Aiming_Game();
    } while((Game_Counter<No_Of_Games)&&(!End_Flag)); /* ESC or all games played */
    nosound();   /* just in case */
    sound(400);
    delay(1000);
    nosound();
    Announce_Session_End();
    getch();  /* show results */
    Close_Graphics();
    return(0);
}
