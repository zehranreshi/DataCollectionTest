# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zehranreshi/DataCollectionTest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zehranreshi/DataCollectionTest/build

# Include any dependencies generated for this target.
include CMakeFiles/jammer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/jammer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/jammer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/jammer.dir/flags.make

CMakeFiles/jammer.dir/src/datacollection.cc.o: CMakeFiles/jammer.dir/flags.make
CMakeFiles/jammer.dir/src/datacollection.cc.o: /home/zehranreshi/DataCollectionTest/src/datacollection.cc
CMakeFiles/jammer.dir/src/datacollection.cc.o: CMakeFiles/jammer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zehranreshi/DataCollectionTest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/jammer.dir/src/datacollection.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/jammer.dir/src/datacollection.cc.o -MF CMakeFiles/jammer.dir/src/datacollection.cc.o.d -o CMakeFiles/jammer.dir/src/datacollection.cc.o -c /home/zehranreshi/DataCollectionTest/src/datacollection.cc

CMakeFiles/jammer.dir/src/datacollection.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/jammer.dir/src/datacollection.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zehranreshi/DataCollectionTest/src/datacollection.cc > CMakeFiles/jammer.dir/src/datacollection.cc.i

CMakeFiles/jammer.dir/src/datacollection.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/jammer.dir/src/datacollection.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zehranreshi/DataCollectionTest/src/datacollection.cc -o CMakeFiles/jammer.dir/src/datacollection.cc.s

# Object files for target jammer
jammer_OBJECTS = \
"CMakeFiles/jammer.dir/src/datacollection.cc.o"

# External object files for target jammer
jammer_EXTERNAL_OBJECTS =

jammer: CMakeFiles/jammer.dir/src/datacollection.cc.o
jammer: CMakeFiles/jammer.dir/build.make
jammer: /usr/lib/x86_64-linux-gnu/libuhd.so
jammer: CMakeFiles/jammer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/zehranreshi/DataCollectionTest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable jammer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jammer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/jammer.dir/build: jammer
.PHONY : CMakeFiles/jammer.dir/build

CMakeFiles/jammer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jammer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jammer.dir/clean

CMakeFiles/jammer.dir/depend:
	cd /home/zehranreshi/DataCollectionTest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zehranreshi/DataCollectionTest /home/zehranreshi/DataCollectionTest /home/zehranreshi/DataCollectionTest/build /home/zehranreshi/DataCollectionTest/build /home/zehranreshi/DataCollectionTest/build/CMakeFiles/jammer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/jammer.dir/depend

