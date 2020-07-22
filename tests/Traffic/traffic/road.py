from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class RoadSegment:
	def __init__(self,vertices):
		self.vertices = vertices
		self.polygon = Polygon(self.vertices)
	def check_in(self,car):
		point = Point(car.position[0],car.position[1])
		return self.polygon.contains(point)

class Road:
	def __init__(self,segments):
		self.segments=segments

	def check_outroad(self,car):
		for segment in self.segments:
			if segment.check_in(car):
				return False
		return True

