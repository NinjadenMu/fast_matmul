#include <assert.h>
#include <stdlib.h>

int get_env_int(const char *env_var, int default_val) {
  const char *val_string = getenv(env_var);
  int val = val_string ? atoi(val_string) : default_val;
  assert(val > 0);
  return val;
}
