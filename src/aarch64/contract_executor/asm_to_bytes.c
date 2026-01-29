#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static void exec_checked(char *const argv[]) {
	execvp(argv[0], argv);
	perror(argv[0]);
	_exit(127);
}

int main(void) {
	int objfd = memfd_create("as-obj", 0);
	if (0 > objfd) die("memfd_create");

	char objpath[64] = { 0 };
	snprintf(objpath, sizeof(objpath), "/proc/self/fd/%d", objfd);

	pid_t pid = fork();
	if (0 == pid) {
		char *as_argv[] = {
			"as",
			"-march=armv9-a+sve+memtag",
			"-o", objpath,
			"-",                 /* read assembly from stdin */
			NULL
		};
		exec_checked(as_argv);
	}

	int status = 0;
	if (0 > waitpid(pid, &status, 0)) die("waitpid");

	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		fprintf(stderr, "as failed\n");
		return 1;
	}

	// rewind ELF
	if (lseek(objfd, 0, SEEK_SET) < 0) die("lseek");

	int outfd = memfd_create("bin-out", 0);
	if (outfd < 0) die("memfd_create outfd");

	char outpath[64];
	snprintf(outpath, sizeof(outpath), "/proc/self/fd/%d", outfd);	

	pid = fork();
	if (0 == pid) {
		char *objcopy_argv[] = {
			"objcopy",
			"-O", "binary",
			objpath,
			outpath,
			NULL
		};
		exec_checked(objcopy_argv);
	}


	if (0 > waitpid(pid, &status, 0)) die("waitpid");

	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		fprintf(stderr, "objcopy failed\n");
		return 1;
	}

	char buf[4096];
	ssize_t n;
	while ((n = read(outfd, buf, sizeof(buf))) > 0) {
		if (write(1, buf, n) != n) die("write stdout");
	}
	if (n < 0) die("read outfd");

	return 0;
}

