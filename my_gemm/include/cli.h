#pragma once

#include <string>
#include <iostream>
#include <getopt.h>

std::string CLI_HELP = 
"\
Usage: main(.exe) [options] file...\r\n\
Options:\r\n\
  --help | -h                               Display this information.\r\n\
  --version | -v                            Display the version.\r\n\
  --shared_mem | -s                         Enalbe shared memory.                                   (Default disable)\r\n\
  --device <integer> | --device=<integer>   Choose device ID.                                       (Default 0)\r\n\
  --HeightA <integer> | --HeightA=<integer> Set the height of matrix A.                             (Default 512)\r\n\
  --WidthA <integer> | --WidthA=<integer>   Set the width of matrix A.                              (Default 512)\r\n\
  --HeightB <integer> | --HeightB=<integer> Set the height of matrix B.                             (Default 512)\r\n\
  --WidthB <integer> | --WidthB=<integer>   Set the width of matrix B.                              (Default 512)\r\n\
  --AllSize <integer> | --AllSize=<integer> Set HeightA, WidthA, HeightB and WidthB in one command. (Default 512)\
";

std::string CLI_VERSION =
"\
My GEMM. 2021-1-22 Version 1.0\
"
;

std::string CLI_INVALID_INFO = 
"\
Invalid command. For more information please type \"--help(-h)\"\
"
;

static struct option CLI_LONG_OPTIONS[] =
{
    {"help",        no_argument,        0,  'h'},
    {"version",     no_argument,        0,  'v'},
    {"shared_mem",  no_argument,        0,  's'},
    {"device",      required_argument,  0,  'd'},
    {"HeightA",     required_argument,  0,  'g'},
    {"WidthA",      required_argument,  0,  'k'},
    {"HeightB",     required_argument,  0,  't'},
    {"WidthB",      required_argument,  0,  'w'},
    {"AllSize",     required_argument,  0,  'a'},
    {0,             0,                  0,  0}
};

static void showHelpInfo()
{
    using namespace std;

    cout << CLI_HELP << endl;
}

static void showVersionInfo()
{
    using namespace std;

    cout << CLI_VERSION << endl;
}

static void showInvalidCLIInfo()
{
    using namespace std;

    cout << CLI_INVALID_INFO << endl;
    showHelpInfo();
}