;+
; NAME:  derot_clio
; PURPOSE:   Derotate a Clio image to put North up and East left
; DESCRIPTION:
;     Input an image and ROTOFF value, it rotates it N up and E left.
;     The Clio instrument rotator north is a parameter that must be
;       measured for each run (Trapezium or other astrometric field)
;       and is written in here for now, until I figure out a better way.
; CATEGORY:
;     Data reduction
; CALLING SEQUENCE:
;     image = derot_clio(image, rotoff)
; EXAMPLE:
;     this_im = derot_clio(im_raw, sxpar(header,'ROTOFF'))
; INPUTS:
;     image -- unrotated image  (or can be cube)
;     rotoff -- ROTOFF from FITS header  (or can be a vector of ROTOFFs for a cube of images)
;     /inverse  --  Set this keyword if the image is already derotated (N up) and you want to re-rotate it (pupil up).
; OUTPUTS:
;     rotated image (N up, E left)
; RESTRICTIONS:
;     The parameter north_CLIO is the instrumental rotator north
;       and is hard-coded below, but should be measured for each run
;       with an astrometric field.
;     This angle gives a counterclockwise rotation.
;     You should multiply it by -1 in IDL using ROT.
; PROCEDURE:
;     Takes in the image, calculates the proper angle, and rotates it with
;     IDL's ROT function.
;     Key equation:
;     angle = rotoff - 180 + north_clio
;     derot_image = rot(image, -1.*angle, cubic=-0.5)
; MODIFICATION HISTORY:
;     Written  2013-03-07  Katie Morzinski   ktmorz@arizona.edu
;     2013-06-03 Added keyword "INVERSE1" that re-rotates derotated images (to facilitate
;                  fake planet insertion for ADI/PCA photometry/astrometry checking)
; BUGS/WISH LIST:
;     Figure out better way than hard-coding to input north_CLIO
;-

FUNCTION derot_clio, image, rotoff, thisdate, inverse1=inverse1
my_name = 'derot_clio'

	;; obsdate -- Comm1
	if n_elements(thisdate) eq 0 then thisdate = '20121201'

	;; Parameter measured from Trapezium astrometry, Comm1, 2012 Nov-Dec.
	caldir = getenv('CLIO_CALDIR')
	readcol,caldir+'north_clio.txt',north_clio,format='(f)',skipline=1
	north_clio = north_clio[0]
	print,north_clio

	if n_elements(size(image,/dim)) le 1 then message,my_name+': Um... image?'

	if n_elements(size(image,/dim)) eq 2 then begin
		;; Single image
		angle = rotoff - 180 + north_clio
		if keyword_set(inverse1) then $
			derot_image = rot(image, angle, cubic=-0.5) else $
			derot_image = rot(image, -1.*angle, cubic=-0.5)

	endif else if n_elements(size(image,/dim)) eq 3 then begin
		;; Cube of images
		for k=0,(size(image))[3]-1 do begin
			angle = rotoff[k] - 180 + north_clio
			if keyword_set(inverse1) then $
				image[*,*,k] = rot(image[*,*,k], angle, cubic=-0.5) else $
				image[*,*,k] = rot(image[*,*,k], -1.*angle, cubic=-0.5)
		endfor;k
		derot_image = image

	endif else begin
		message,my_name+': Um... image?'
	endelse
	
RETURN,derot_image
END
