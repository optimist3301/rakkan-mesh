package obs

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Location claims (spec §9). A position is one claim:
//
//	loc.geo = "lat,lon,radius_m"   WGS84 degrees; radius = ~95% error circle
//
// loc.logical (free text, "site/building/floor/room") and loc.region (a
// country or city name) carry places that have no coordinates.

const maxRadiusM = 20_000_000

// FormatGeo renders the canonical value of a loc.geo claim.
func FormatGeo(lat, lon, radiusM float64) string {
	return fmt.Sprintf("%.6f,%.6f,%.1f", lat, lon, radiusM)
}

// ParseGeo parses and validates a loc.geo value.
func ParseGeo(s string) (lat, lon, radiusM float64, err error) {
	parts := strings.Split(s, ",")
	if len(parts) != 3 {
		return 0, 0, 0, errors.New("want lat,lon,radius")
	}
	var v [3]float64
	for i, p := range parts {
		if v[i], err = strconv.ParseFloat(strings.TrimSpace(p), 64); err != nil || math.IsNaN(v[i]) || math.IsInf(v[i], 0) {
			return 0, 0, 0, fmt.Errorf("bad number %q", p)
		}
	}
	lat, lon, radiusM = v[0], v[1], v[2]
	switch {
	case lat < -90 || lat > 90:
		return 0, 0, 0, errors.New("latitude out of range")
	case lon < -180 || lon > 180:
		return 0, 0, 0, errors.New("longitude out of range")
	case radiusM <= 0 || radiusM > maxRadiusM:
		return 0, 0, 0, errors.New("radius out of range")
	}
	return lat, lon, radiusM, nil
}
