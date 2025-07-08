import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const PHOSPHO_PRO_SUBSCRIBE_URL =
  "https://billing.phospho.ai/b/5kQbJ2grJcfv4b369Y33W0g";

export function PhosphoProCallout({ className }: { className?: string }) {
  return (
    <Card className={cn("border-green-400", className)}>
      <CardContent className="flex items-center p-6">
        <div className="flex-1">
          <div className="font-semibold text-lg mb-1.5">
            Boost your phospho experience with{" "}
            <span className="text-green-500">phospho pro</span>
          </div>
          <div className="mb-3 text-muted-foreground">
            Access the phospho teleoperation app, AI training, exclusive Discord
            channels and more.
          </div>
          <Button asChild>
            <a
              href={PHOSPHO_PRO_SUBSCRIBE_URL}
              target="_blank"
              rel="noopener noreferrer"
            >
              Subscribe to phospho pro
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default PhosphoProCallout;
