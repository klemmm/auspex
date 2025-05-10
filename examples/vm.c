#include <stdio.h>

#define MOVI 0
#define JMPNZ 1
#define ADD 2
#define HALT 3

int code[21];

int regs[8];

int main(void) {
	int vip = 0;
	int running = 1;

    /* Done like this because AUSPEX doesn't yet support reading from .data / .rodata for initial state */
	code[0] = MOVI;
	code[1] = 0;
	code[2] = 20;

    code[3] = JMPNZ;
	code[4] = 2;
	code[5] = 6;

    code[6] = MOVI;
	code[7] = 1;
	code[8] = 0;

	code[9] = ADD;
	code[10] = 0;
	code[11] = -1;

    code[12] = ADD;
	code[13] = 1;
	code[14] = 1;

	code[15] = JMPNZ;
	code[16] = 0;
	code[17] = 2;

	code[18] = HALT;
	code[19] = 0;
	code[20] = 0;


	while(running) {
        int opcode = code[vip];
        int operand1 = code[vip + 1];
        int operand2 = code[vip + 2];
		vip += 3;

		if (opcode == HALT) {
			running = 0;
            printf("halting\n");
		} else if (opcode == MOVI) {
			regs[operand1] = operand2;
		} else if (opcode == ADD) {
			printf("adding\n");
			regs[operand1] += operand2;
		} else if (opcode == JMPNZ) {
			if (regs[operand1] != 0) {
				vip = 3*(operand2);
			}
		}
	}
	printf("finished\n");
}
