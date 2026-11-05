"""R-RTSP wire protocol helpers.

The server deliberately keeps the wire format independent from the model
input contract.  RTP carries source observations; model inference is exposed
through ``core.server.model`` instead.
"""

from .auth import DigestAuthenticator
from .payloads import (
    CLASS_TO_ID,
    RD_HEADER,
    TRACK_FIELDS,
    TRACK_STATE_NAMES,
    build_inference_frame,
    build_track_frame,
    encode_inference_result,
    encode_track,
    fragment_rd_frame,
)
from .rtp import (
    RTP_HEADER,
    build_receiver_report,
    build_rtp_packet,
    build_sender_report,
    parse_rtcp,
    parse_rtp_packet,
)
from .rtsp import (
    DEFAULT_PORT,
    RESOURCE,
    Request,
    build_sdp,
    media_track_id,
    parse_transport,
    read_request,
    send_response,
)

__all__ = [
    "CLASS_TO_ID",
    "DEFAULT_PORT",
    "DigestAuthenticator",
    "RD_HEADER",
    "RESOURCE",
    "RTP_HEADER",
    "Request",
    "TRACK_FIELDS",
    "TRACK_STATE_NAMES",
    "build_receiver_report",
    "build_inference_frame",
    "build_rtp_packet",
    "build_sdp",
    "build_sender_report",
    "build_track_frame",
    "encode_track",
    "encode_inference_result",
    "fragment_rd_frame",
    "media_track_id",
    "parse_rtcp",
    "parse_rtp_packet",
    "parse_transport",
    "read_request",
    "send_response",
]
