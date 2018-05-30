CC=clang
LIBPATH=../gym_space_fortress/envs/space_fortress/shared
CREATE_BINARY=no


# Uncomment to specify parameters
PARAM=
PARAM+=-D GUI_INTERFACE #** Full sized and colored game renders **
#PARAM+=-D GRID_MOVEMENT #** Lowers the control order to a direct type of control **
#PARAM+=-D NO_WRAP #** Turns off wrapping **
#PARAM+=-D NO_DIRECTION #** Turns off movement based on the ships nose direction **
#PARAM+=-D DEBUG #** Sounds Effects/Printing messages on soundless linux **
#PARAM+=-D ROTATE_ANGLE=theta #** The rotation of the ship in degrees **
#PARAM+=-D NO_RANDOM_SPAWN #** Disable random spawn location and random ship orientation **

define build
	$(CC) -march=native `pkg-config --cflags cairo` -Wall -g -fPIC -c $(1)/myvars.c $(1)/DE_Minimal.c $(1)/HM.c $(1)/TCOL.c $(1)/RS.c -Wno-dangling-else -Wno-switch -O3 $(PARAM);
	$(CC) -march=native `pkg-config --cflags cairo --libs cairo` -shared -o $(2)_frame_lib.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 $(PARAM);
	#$(CC) -march=native `pkg-config --cflags cairo --libs cairo` -shared -o $(2)_frame_lib.so myvars.o HM.o RS.o TCOL.o DE_Minimal.o -O3 $(PARAM);\
	
	@# check if we have to build the binary
	if [ $(CREATE_BINARY) = yes ];then\
		$(CC) -march=native `pkg-config --cflags cairo pkg-config --libs cairo pkg-config --cflags gtk+-3.0 pkg-config --libs gtk+-3.0 ` -Wall -g -v -fPIC $(1)/myvars.c $(1)/TCOL.c $(1)/DE_Minimal.c $(1)/HM.c $(1)/RS.c  -lm -o Control -Wno-dangling-else -Wno-switch $(GUI);\
	fi
	
	#mv *.o $(1)
	rm *.o
	mv *.so $(LIBPATH)
	@echo "Shared library $(2)_frame_lib.so sucessfully made and installed."
endef

AIM Aim aim:
	$(call build, AIM_Cairo, aim)
SF Sf sf:
	$(call build, SF_Cairo, sf)
SFC Sfc sfc:
	$(call build, SFC_Cairo, sfc)

ALL All all: AIM SF SFC

CLEAN Clean clean:
	@rm -f *.o *.so AIM_Cairo/*.o SF_Cairo/*.o SFC_Cairo/*.o $(LIBPATH)/*.so
