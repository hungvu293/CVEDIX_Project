#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <mqtt/async_client.h>

class Message : public virtual mqtt::callback {
public:
    Message(const std::string& server_address, const std::string& client_id);
    ~Message();

    void connect();
    void disconnect();
    void sendMessage(const std::chrono::system_clock::time_point& capture_time, int camera_id, const cv::Mat& frame);

private:
    void connected(const std::string& cause) override;
    void connection_lost(const std::string& cause) override;
    void message_arrived(mqtt::const_message_ptr msg) override;
    void delivery_complete(mqtt::delivery_token_ptr token) override;

    mqtt::async_client client;
    mqtt::connect_options connOpts;
    std::string topic = "data/plates";
    const int QOS = 1;
};

#endif // MESSAGE_H
