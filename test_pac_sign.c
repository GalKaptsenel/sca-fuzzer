  #include <fcntl.h>
  #include <sys/ioctl.h>
  #include <stdint.h>
  #include <stdio.h>
  #include <string.h>
#include <unistd.h>

  struct pac_sign_req { uint64_t ptr; uint64_t ctx; char mnemonic[16]; uint64_t result; };
  #define REVISOR_PAC_SIGN _IOWR('r', 12, struct pac_sign_req)
  int main() {
      int fd = open("/dev/executor", O_RDWR);
      if (fd < 0) { perror("open"); return 1; }
      struct pac_sign_req req = { .ptr = 0x0000ffff12345678ULL, .ctx = 0x42ULL };
      strcpy(req.mnemonic, "pacia");
      ioctl(fd, REVISOR_PAC_SIGN, &req);
      printf("pacia(0x%016llx, 0x42) = 0x%016llx\n", req.ptr, req.result);
      // Signed value should differ from input and have PAC bits in top byte
      printf("%s\n", req.result != req.ptr ? "PASS: value was signed" : "FAIL: result unchanged");
      close(fd);
  }

