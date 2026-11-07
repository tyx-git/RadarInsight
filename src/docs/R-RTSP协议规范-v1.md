# R-RTSP v1 协议规范

> 名称：**R-RTSP（Radar Real-Time Streaming Protocol）**  
> 状态：v1 已实现  
> 原则：**信令走 TCP，媒体走 UDP**。  
> 默认信令端口：`8554`  
> 默认资源路径：`/radar`

R-RTSP 用 RTSP 文本命令完成会话控制，用 RTP over UDP 发送雷达 RD 图谱和航迹更新。它的使用体验接近网络摄像头：服务端启动后，接收端连接同一个 URL，随后持续接收数据。

## 1. URL

```text
rtsp://<服务器IP>:8554/radar
```

示例：

```text
rtsp://127.0.0.1:8554/radar
rtsp://<服务器IP>:8554/radar
```

## 2. 协议组成

| 部分 | 承载 | 用途 |
| --- | --- | --- |
| RTSP | TCP | `OPTIONS/DESCRIBE/SETUP/PLAY/PAUSE/TEARDOWN/GET_PARAMETER/SET_PARAMETER` |
| SDP | RTSP 响应体 | 声明媒体、维度、载荷参数 |
| RTP | UDP | 雷达数据载荷，PT=96/97；可选分类结果 PT=98 |
| RTCP | UDP | Sender Report、Receiver Report、统计反馈 |
| Digest | RTSP 头部 | 可选 L1 鉴权 |

## 3. RTSP 方法与状态机

| 方法 | 用途 |
| --- | --- |
| `OPTIONS` | 查询服务器支持的方法 |
| `DESCRIBE` | 获取 SDP |
| `SETUP` | 协商客户端/服务端 UDP 端口和 SSRC |
| `PLAY` | 开始发送 RTP |
| `PAUSE` | 暂停发送 RTP |
| `TEARDOWN` | 关闭会话并释放 UDP 端口 |
| `GET_PARAMETER` | 查询事件数量、当前会话发送周期等参数 |
| `SET_PARAMETER` | 设置当前会话的 `send_interval_seconds`（RadarInsight 后端支持） |

### 3.1 发送周期设置

RadarInsight `core.server` 支持按会话设置发送周期，单位为秒。客户端在
`SETUP` 后、`PLAY` 前发送以下请求，`Session` 使用 `SETUP` 返回的会话编号：

```text
SET_PARAMETER rtsp://host:8554/radar RTSP/1.0
CSeq: 4
Session: <session-id>
Content-Type: text/parameters
Content-Length: 29

send_interval_seconds: 1.25
```

正文以 `\r\n` 结尾。`send_interval_seconds` 必须是有限的正数，可包含小数。
服务端成功时返回 `200`，响应体按相同格式返回实际生效值；参数无效返回 `400`，
会话无效返回 `454`。`GET_PARAMETER` 返回当前会话的实际周期。

此设置仅影响该会话，不修改其他客户端的发送周期或服务端启动默认值。
首次 `DESCRIBE` 的 SDP `updateRate` 表示服务端默认频率（Hz）；设置成功后，
该会话的实际频率为 `1 / send_interval_seconds`。

AeroSight 的“数据源 → 后端发送周期”应用时会重连该实例并设置新周期；
“显示 → 前端扫描周期”只控制本地扫描动画，两项均以 `s` 输入，互不覆盖。
推理或队列补充未完成时，实际发送可能晚于配置周期；不突发补发积压批次。
旧服务端若不支持设置，客户端会明确报告设置失败。

状态：

```text
INIT --DESCRIBE--> READY
READY --SETUP--> READY
READY --PLAY--> PLAYING
PLAYING --PAUSE--> READY
PLAYING/READY --TEARDOWN--> INIT
```

## 4. 典型时序

```text
C -> S : OPTIONS rtsp://host:8554/radar
S -> C : 200 OK

C -> S : DESCRIBE rtsp://host:8554/radar
S -> C : 200 OK + SDP

C -> S : SETUP rtsp://host:8554/radar/trackID=1
         Transport: RTP/AVP;unicast;client_port=5004-5005
S -> C : 200 OK
         Transport: RTP/AVP;unicast;client_port=5004-5005;
                    server_port=6200-6201;ssrc=...

C -> S : PLAY rtsp://host:8554/radar
S -> C : 200 OK
S -> C : RTP packets over UDP

C -> S : TEARDOWN rtsp://host:8554/radar
S -> C : 200 OK
```

## 5. SDP

RD 数据集包含两条媒体：`trackID=0` 为 RD 图谱，`trackID=1` 为航迹更新。
启用服务端分类推理时，可选增加 `trackID=2` / `PT=98`；客户端可尝试 SETUP，
返回 `461` 表示服务端未提供该通道。纯航迹数据集只包含 `trackID=1`。

示例：

```sdp
v=0
o=radar 123456 1 IN IP4 127.0.0.1
s=R-RTSP Radar Stream
t=0 0
a=control:*
a=range:npt=0-
m=application 5004 RTP/AVP 96
a=control:trackID=0
a=rtpmap:96 radar-rdmap/90000
a=fmtp:96 numRangeBins=31;numDopplerBins=226;bitsPerSample=32;quant=linear;compression=none;frameRate=5;maxPacketSize=1200;samples=amplitude,phase
a=ssrc:00112233 cname:radar@aerosight
m=application 5006 RTP/AVP 97
a=control:trackID=1
a=rtpmap:97 radar-track/90000
a=fmtp:97 encoding=tlv;updateRate=5;coord=polar;fields=range,azimuth,elevation,radial_speed,speed,course,snr,amplitude,class
a=ssrc:44556677 cname:radar@aerosight
```

> 上例中的 numRangeBins/numDopplerBins 仅为示例；实际维度由数据集决定，以每帧 RD 头部为准，同一会话内也可能变化（当前数据集出现过 31×226、31×360、31×570、31×900）。


### 5.1 角度约定

航迹 TLV 中的 `azimuth_deg` 与 `course_deg` 使用有符号角度：

- 单位：度；
- `0°`：雷达正前方 / 当前速度方向；
- 正数：顺时针，即雷达右侧；
- 负数：逆时针，即雷达左侧；
- 范围：`-180°~180°`。

客户端直角坐标约定为 `x=前方、y=左方、z=上方`。球坐标转换关系为：

```text
x = range * cos(elevation) * cos(azimuth)
y = -range * cos(elevation) * sin(azimuth)
z = range * sin(elevation)
```

## 6. RTP 固定头部

使用 RFC 3550 RTP fixed header：

| 字段 | 长度 | v1 取值 |
| --- | --- | --- |
| Version | 2 bit | 2 |
| Padding | 1 bit | 0 |
| Extension | 1 bit | 0 |
| CSRC Count | 4 bit | 0 |
| Marker | 1 bit | RD 帧最后一个分片置 1 |
| Payload Type | 7 bit | 96 或 97；可选 98 |
| Sequence Number | 16 bit | 每个 RTP 包递增 |
| Timestamp | 32 bit | 90 kHz 时钟 |
| SSRC | 32 bit | SETUP 时声明 |

## 7. PT=96：RD 图谱载荷

一帧 RD 图由一个或多个 RTP 包承载。每个 RTP payload 都携带完整 20 字节 RD 头部，随后是矩阵字节。

### 7.1 RD 头部

所有头部字段使用大端序：

| 字段 | 类型 | 字节 |
| --- | --- | --- |
| frameType | uint8 | 1 |
| flags | uint8 | 1 |
| frameSeq | uint16 | 2 |
| frameId | uint32 | 4 |
| numRangeBins | uint16 | 2 |
| numDopplerBins | uint16 | 2 |
| bitsPerSample | uint8 | 1 |
| quant | uint8 | 1 |
| compr | uint16 | 2 |
| scale0 | float32 | 4 |

共 **20 字节**。

### 7.2 flags

| 位 | 含义 |
| --- | --- |
| bit0 | 保留，ROI 扩展 |
| bit1 | 1=包含相位矩阵 |
| bit2 | 1=样本为小端 float32 |

当前 v1：`flags=6`，表示小端样本且包含相位。

### 7.3 样本布局

- 矩阵维度：不固定，以每帧头部 numRangeBins × numDopplerBins 为准（当前数据集出现过 `31×226`、`31×360`、`31×570`、`31×900`）
- 排列：range-major，行优先
- 样本：float32 little-endian
- 当存在相位时，payload 顺序为：

```text
amplitude[R * D] + phase[R * D]
```

单帧原始样本大小：

```text
2 × R × D × 4 字节（例如 31×226 时为 56,048 字节）
```

### 7.4 分片

默认 `maxPacketSize=1200`，即每个 RTP payload 尽量不超过 1200 字节。接收端根据：

1. 相同 `frameSeq`
2. RTP sequence number
3. 最后一包 RTP Marker=1

完成一帧重组。

## 8. PT=97：航迹 TLV 载荷

一个 RTP 包承载一个更新周期内的航迹集合。

### 8.1 航迹帧头

大端序：

| 字段 | 类型 |
| --- | --- |
| frameType | uint8 |
| numTracks | uint16 |
| sequence | uint8 |
| reserved | uint8 |

共 5 字节。

### 8.2 Track TLV

每个航迹为一个 TLV：

```text
type(uint8) + length(uint16) + value(length bytes)
```

v1 中 `type=1`。

Value 布局：

| 字段 | 类型 |
| --- | --- |
| trackId | uint32 |
| state | uint8 |
| classId | uint8 |
| FSPEC | uint32 |
| fields | 按 FSPEC 出现，每个为 float32 大端序 |

### 8.3 state

| 值 | 含义 |
| --- | --- |
| 0 | 疑似 |
| 1 | 确认 |
| 2 | 外推 |
| 3 | 消亡 |

### 8.4 classId

| 值 | 含义 |
| --- | --- |
| 1 | 无人机 |
| 2 | 鸟 |
| 3 | 气球 |
| 4 | 杂波 |
| 5 | 未知/待确认 |
| 6 | 其他 |

### 8.5 FSPEC 字段位

| FSPEC 位 | 字段 |
| --- | --- |
| bit0 | range_m |
| bit1 | azimuth_deg |
| bit2 | elevation_deg |
| bit3 | radial_speed_mps |
| bit4 | speed_mps |
| bit5 | course_deg |
| bit6 | snr_db |
| bit7 | amplitude_db |

只有 FSPEC 置位的字段才会出现在 TLV 中。

## 9. PT=98：可选模型分类结果

启用服务端推理时，SDP 可额外声明 `trackID=2`、`PT=98`。
该通道是 RadarInsight 分类结果扩展，不属于雷达原始观测；未声明时，
接收端不得假设存在该通道。服务端在对应 PT=97 航迹更新之前发送结果，
两者 RTP timestamp 和 `trackId` 相同。

Payload 使用与 PT=97 相同的 5 字节 frame header，`frameType=2`。每个结果
使用 `TLV type=2`，Value 固定 9 字节，大端序：

| 字段 | 类型 | 字节 |
| --- | --- | --- |
| trackId | uint32 | 4 |
| classId | uint8 | 1 |
| confidence | float32 | 4 |

`classId` 映射为 `1=drone`、`2=bird`、`3=balloon`、`4=clutter`、
`5=UNKNOWN_REJECTED`、`6=PENDING_ABSTAIN`。`confidence` 必须是有限 `[0,1]`
数值。PT=97 的 `classId` 是数据源类别，不得用 PT=98 结果覆盖它。

## 10. RTCP

v1 实现可互操作的最小集合：

- 服务端周期发送 `Sender Report`
- 接收端周期发送 `Receiver Report`
- 保留 `APP` 包构造/解析能力

RTCP 端口为 RTP 端口 +1，由 SETUP 协商。

## 11. 鉴权

### 11.1 模式

| 等级 | 名称 | 使用方式 |
| --- | --- | --- |
| L0 | 开放模式 | 不需要账号信息 |
| L1 | 私密模式 | Digest SHA-256 挑战应答 |

### 11.2 开放模式

```text
rtsp://<服务器IP>:<端口>/radar
```

### 11.3 私密模式

账号信息由使用者在推流工具中自行设置。普通链接格式：

```text
rtsp://<服务器IP>:<端口>/radar
```

如接收端支持 URL 账号信息，也可使用：

```text
rtsp://<用户名>:<密码>@<服务器IP>:<端口>/radar
```

该方式不会加密链接，请妥善保管。

鉴权流程：

```text
接收端 -> DESCRIBE/SETUP/PLAY
服务端 -> 401 Unauthorized
接收端 -> Authorization: Digest ...
服务端 -> 校验通过后返回 200 OK
```

哈希计算：

```text
HA1 = SHA256(username:realm:password)
HA2 = SHA256(method:uri)
response = SHA256(HA1:nonce:nc:cnonce:qop:HA2)
```

## 12. 与 .asd 数据集的关系

`.asd` 是单文件数据集容器；R-RTSP 是网络传输协议。推流工具读取 `.asd`，并将其中事件转换为 RTP 数据包。

```text
.asd 文件 -> R-RTSP 服务端 -> RTSP/SDP 建会话 -> UDP/RTP 数据 -> 接收端
```
