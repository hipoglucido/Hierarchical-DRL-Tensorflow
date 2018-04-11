#include <stdio.h>
#include <ctype.h>
#include <string.h>
//#include <dir.h>
//#include <dos.h>
//#include <conio.h>
//#include <graphics.h>
#include <stdlib.h>
#include <time.h>

#include "myconst.h"
#include "myext.h"  /* modify to myext.h */

//int readkey(void);
//int write_file(char filename[],int first_time,int sess);
//int read_file(char filename[],int displ,int sess);
//int check_id(void);
//void user_dialogue(void);
//int ask_training_session(void);
//void opening_screen(void);
//int ask_session(void);
//int ask_game(void);
//void displ_screen(int scr_no,char *scr_file);
//char *ask_session_nr(char filename[MAX_DIR_PATH],char string[10]);
//void final_display(void);
//void erase_files(char path1[]);
//void open_graphics(void);

/**************************************************************************/
/*                                                                        */
/*                            MAIN PROGRAM                                */
/*                                                                        */
/**************************************************************************/


int main() {
//    Open_Graphics();
//    Set_Bonus_Message();
//    Set_Bonus_Chars();
//    Set_Graphics_Eraser();
//    Close_Graphics();

//    randomize(); // I think I added this
//    delay(0); // Nice delay of 0 seconds
		usleep(5);

    Run_SF();

    return;
}

/**************************************************************************/
/*                                                                        */
/*  final display - Displays the screen with all the game results.        */
/*  Called by Run Session                                                 */
/*                                                                        */
/**************************************************************************/

void final_display() {
//    clrscr(); // From conio.h, clears the MS-DOS screen
//    displ_screen(4,"screen.dat");
//    gotoxy(44,8);printf("%d",Score);
//    gotoxy(39,12);printf("%d",Ship_Damaged_By_Fortress);
//    gotoxy(42,14);printf("%d",Ship_Damaged_By_Mines);
//    gotoxy(50,16);printf("%d",Fortress_Destroyed);
//    gotoxy(11,21);printf("%d",Points);
//    gotoxy(30,21);printf("%d",Control);
//    gotoxy(50,21);printf("%d",Velocity);
//    gotoxy(67,21);printf("%d",Speed);
//    gotoxy(27,23);printf("%d",No_Of_Bonus_Intervals);
//    gotoxy(48,23);printf("%d",No_Of_Points_Bonus_Taken);
//    gotoxy(70,23);printf("%d",No_Of_Missiles_Bonus_Taken);
}

/**************************************************************************/
/*                                                                        */
/* displ_screen Reads the file"SCREEN.DAT"                                */
/* Parameters:                                                            */
/*    scr_no  -number of screen to be read from the file                  */
/*    scr_file -file name("SCREEN.DAT")                                   */
/*                                                                        */
/**************************************************************************/

// We don't display the score
void displ_screen(int scr_no,char *scr_file) {
//    FILE *f;
//    int ch,i;
//    if(((f=fopen(scr_file,"r"))==NULL)) {
//        printf("Cannot open %s\n",scr_file);
//        getch();
//        return;
//    }
//    i=0;
//    while(i<scr_no)
//        if ((ch=getc(f))=='@') i++;
//    while (((ch=getc(f))!='@')&&(ch!=EOF))
//        putchar(ch);
//    fclose(f);
}

/**************************************************************************/
/*                                                                        */
/*  open_graphics  - Opens the graphic mode                               */
/*  Called by opening_screen                                              */
/*                                                                        */
/**************************************************************************/

//void open_graphics(void) {
//    int g_driver,g_mode,g_error;
//
//    g_driver=DETECT;
//    initgraph(&g_driver,&g_mode,"C:\\BORLANDC\\BGI");
//    g_error=graphresult();
//    if(g_error<0) {
//        printf("INITGRAPH ERROR : %s.\n",grapherrormsg(g_error));
//        exit(1);
//    }
}
