#pragma once

#include <string>
#include <iostream>
#include <getopt.h>

std::string CLI_HELP = 
"\
Usage: main(.exe) [options] file...\r\n\
Options:\r\n\
  --help                Display this information.\r\n\
  --version             Display the version.\r\n\
  --sharedmem           Use shared memory.\r\n\
  --device  <integer>   Choose device ID.\r\n\
  --HeightA <integer>   Set the height of matrix A.\r\n\
  --WidthA  <integer>   Set the width of matrix A.\r\n\
  --HeightB <integer>   Set the height of matrix B.\r\n\
  --WidthB  <integer>   Set the width of matrix B.\
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
    {"device",      required_argument,  0,  'd'},
    {"sharedmem",   required_argument,  0,  's'},
    {"HeightA",     required_argument,  0,  'g'},
    {"WidthA",      required_argument,  0,  'k'},
    {"HeightB",     required_argument,  0,  't'},
    {"WidthB",      required_argument,  0,  'w'},
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