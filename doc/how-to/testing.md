# Test a documented workflow

Configure a Debug build with GoogleTest and application testing enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_GOOGLE_TESTING=ON -DENABLE_DEAL_II_APP_TESTING=ON
cmake --build build -j
```

To compile and test both single-config variants in one tree, configure with
`-DCMAKE_BUILD_TYPE=DebugRelease`. Release test executables remain unsuffixed
and Debug executables use the `_debug` suffix; CTest gives the two variants
distinct names.

Run the application smoke tests from the build tree:

```bash
(cd build && ./gtests/gtests_debug --gtest_filter='AppExecutables.*')
```

The smoke tests launch applications from private build-tree directories and
use the same configured tutorial inputs referenced by the prose. For the full
validation matrix, see [Developer testing](../developer/testing).
