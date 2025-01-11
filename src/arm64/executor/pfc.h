#ifndef ARM64_EXECUTOR_PFC_H
#define ARM64_EXECUTOR_PFC_H

#include "main.h"

struct pfc_config {
    unsigned long evt_num;
    unsigned long umask;
    unsigned long cmask;
    unsigned int any;
    unsigned int edge;
    unsigned int inv;
};


#endif // ARM64_EXECUTOR_PFC_H
