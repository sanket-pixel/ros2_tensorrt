. tools/environment.sh

mkdir -p build

cd build
cmake ..
make > /dev/null
./main --$MODE