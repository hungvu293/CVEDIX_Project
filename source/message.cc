#include "message.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>

// ---- Base64 Encoding ----
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static std::string base64_encode(const std::vector<uchar>& buf) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    for (const auto& byte : buf) {
        char_array_3[i++] = byte;
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }

    return ret;
}
// ---- End Base64 ----

Message::Message(const std::string& server_address, const std::string& client_id)
    : client(server_address, client_id) {
    connOpts.set_keep_alive_interval(20);
    connOpts.set_clean_session(true);
    client.set_callback(*this);
}

Message::~Message() {
    disconnect();
}

void Message::connect() {
    try {
        std::cout << "Connecting to MQTT broker..." << std::endl;
        client.connect(connOpts)->wait();
    } catch (const mqtt::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
    }
}

void Message::disconnect() {
    if (client.is_connected()) {
        std::cout << "Disconnecting from MQTT broker..." << std::endl;
        client.disconnect()->wait();
    }
}

void Message::sendMessage(const std::chrono::system_clock::time_point& capture_time, int camera_id, const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(sendMutex);
    
    if (!client.is_connected()) {
        std::cerr << "MQTT client not connected. Cannot send message." << std::endl;
        return;
    }

    // 1. Encode frame to JPEG format in memory
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", frame, buf)) {
        std::cerr << "Failed to encode frame to JPEG" << std::endl;
        return;
    }

    // 2. Encode JPEG buffer to Base64
    std::string base64_frame = base64_encode(buf);
    std::string data_uri_frame = "data:image/jpeg;base64," + base64_frame;

    // 3. Format timestamp
    auto time_t = std::chrono::system_clock::to_time_t(capture_time);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(capture_time.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
    std::string timestamp_str = ss.str();

    // 4. Create JSON payload
    std::string payload = "{\"camera_id\": " + std::to_string(camera_id) + ", "
                          "\"capture_time\": \"" + timestamp_str + "\", "
                          "\"frame\": \"" + data_uri_frame + "\"}";

    // 5. Publish message
    try {
        std::cout << "send msg" << std::endl;
        mqtt::message_ptr pubmsg = mqtt::make_message(topic, payload);
        pubmsg->set_qos(QOS);
        client.publish(pubmsg);
    } catch (const mqtt::exception& exc) {
        std::cerr << "Error publishing message: " << exc.what() << std::endl;
    }
}

void Message::connected(const std::string& cause) {
    std::cout << "Connection success" << std::endl;
}

void Message::connection_lost(const std::string& cause) {
    std::cerr << "Connection lost: " << cause << std::endl;
}

void Message::message_arrived(mqtt::const_message_ptr msg) {
    // Not used for publishing client
}

void Message::delivery_complete(mqtt::delivery_token_ptr token) {
    // std::cout << "Delivery complete for token: " << (token ? token->get_message_id() : 0) << std::endl;
}
