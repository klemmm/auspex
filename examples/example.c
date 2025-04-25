int main(int argc, char **argv) {
    unsigned int i;
    unsigned int ret = 0;
    for (i = 0; i < argc; i++) {
        if (argv[i][0] != 0) {
            ret++;
        }
    }
    return ret;
}

