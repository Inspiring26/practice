function FindProxyForURL(url, host) {
  return "PROXY 192.168.1.80:1080; SOCKS5 192.168.1.80:1080; SOCKS 192.168.1.80:1080;";
}
