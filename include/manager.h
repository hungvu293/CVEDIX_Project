#include <vector>
#include <thread>

#include "manager.h"
#include "data.h"
#include "yolov8.h"
#include "track.h"
#include "reader.h"
#include "osd.h"
// #include "mqtt"

class Pipeline {
private:
    std::vector<std::unique_ptr<Reader>, 2> readers;
    
}