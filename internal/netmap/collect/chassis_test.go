//go:build linux

package collect

import (
	"os"
	"path/filepath"
	"testing"
)

func TestChassisClass(t *testing.T) {
	sys := func(files map[string]string) string {
		root := t.TempDir()
		for p, v := range files {
			full := filepath.Join(root, p)
			os.MkdirAll(filepath.Dir(full), 0o755)
			os.WriteFile(full, []byte(v+"\n"), 0o644)
		}
		return root
	}
	cases := []struct {
		files map[string]string
		want  string
	}{
		{map[string]string{"class/dmi/id/chassis_type": "10"}, "laptop"},
		{map[string]string{"class/dmi/id/chassis_type": "3"}, "desktop"},
		{map[string]string{"class/dmi/id/chassis_type": "23"}, "server"},
		{map[string]string{"class/dmi/id/chassis_type": "30"}, "tablet"},
		{map[string]string{"class/dmi/id/chassis_type": "1", "class/dmi/id/sys_vendor": "QEMU"}, "server"},
		// "Other" chassis, but a system battery
		{map[string]string{"class/dmi/id/chassis_type": "1", "class/power_supply/BAT0/type": "Battery",
			"class/power_supply/BAT0/scope": "System"}, "laptop"},
		// a wireless mouse's battery is not the machine's
		{map[string]string{"class/dmi/id/chassis_type": "2", "class/power_supply/hidpp_battery_0/type": "Battery",
			"class/power_supply/hidpp_battery_0/scope": "Device"}, ""},
		{map[string]string{}, ""},
	}
	for i, c := range cases {
		if got := chassisClass(sys(c.files)); got != c.want {
			t.Errorf("case %d: got %q, want %q", i, got, c.want)
		}
	}
}
