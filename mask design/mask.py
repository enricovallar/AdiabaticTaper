
import nazca as nd

nd.strt(length=20).put()
nd.bend(angle=90).put()
nd.bend(angle=-180).put()
nd.strt(length=20).put()

nd.export_gds("mask.gds")