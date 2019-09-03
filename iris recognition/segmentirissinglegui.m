% segmentiris - peforms automatic segmentation of the iris region
% from an eye image. Also isolates noise areas such as occluding
% eyelids and eyelashes.
%
% Usage: 
% [circleiris, circlepupil, imagewithnoise] = segmentiris(image)
%
% Arguments:
%	eyeimage		- the input eye image
%	
% Output:
%	circleiris	    - centre coordinates and radius
%			          of the detected iris boundary
%	circlepupil	    - centre coordinates and radius
%			          of the detected pupil boundary
%	imagewithnoise	- original eye image, but with
%			          location of noise marked with
%			          NaN values
%
% Author: 
% Libor Masek
% masekl01@csse.uwa.edu.au
% School of Computer Science & Software Engineering
% The University of Western Australia
% November 2003

%eyeimage='C:/Users/Lenovo/Downloads/01_L.bmp'
%function [circleiris, circlepupil, imagewithnoise] = segmentiris(eyeimage)

%CASIA
function [c,dddd,circleiris, circlepupil, imagewithnoise]= segmentiris(eyeimage)
    lpupilradius = 28;
    upupilradius = 75;
    lirisradius = 80;
    uirisradius = 150;
    eyeimage=imread(eyeimage)
    %disp(size(eyeimage))

    c=zeros(1,3);
    
    dddd=zeros(1,3);
    disp(class(dddd))
    disp(c)
% define scaling factor to speed up Hough transform
    scaling = 0.4;
    reflecthres = 240;


    % find the iris boundary by Daugman's intefrodifferential operaror
    [rowp, colp, rp] = SearchInnerBoundary(double(eyeimage));
    disp('jeee')
    disp(rowp)
    [row, col, r] = SearchOuterBoundary(double(eyeimage), rowp, colp, rp) ;
    circleiris = [row, col, r];
    c=[c,circleiris]
    disp(c)
    rowd = double(row);
    cold = double(col);
    rd = double(r);

    irl = round(rowd-rd);
    iru = round(rowd+rd);
    icl = round(cold-rd);
    icu = round(cold+rd);

    imgsize = size(eyeimage);
        irl = 1;
    

    if icl < 1
        icl = 1;
    end

    if irl < 1 
    end

    if iru > imgsize(1)
        iru = imgsize(1);
    end

    if icu > imgsize(2)
        icu = imgsize(2);
    end

    % to find the inner pupil, use just the region within the previously
    % detected iris boundary
    imagepupil = double(eyeimage( irl:iru,icl:icu));

    rowp = double(rowp)-irl;
    colp = double(colp)-icl;
    r = double(rp);

    row = double(irl) + rowp;
    col = double(icl) + colp;

    row = round(row);
    col = round(col);

    circlepupil = [row col r];
    dddd=[dddd,circlepupil]
    disp(dddd)


    % set up array for recording noise regions
    % noise pixels will have NaN values
    imagewithnoise = double(eyeimage);

    %find top eyelid
    topeyelid = uint8(imagepupil(1:(rowp-r),:));
    lines = findline(topeyelid);

    if size(lines,1) > 0
        [xl yl] = linecoords(lines, size(topeyelid));
        yl = double(yl) + irl-1;
        xl = double(xl) + icl-1;

        yla = max(yl);

        y2 = 1:yla;

        ind3 = sub2ind(size(eyeimage),yl,xl);
        imagewithnoise(uint32(ind3)) = NaN;   
        imagewithnoise(round(y2), round(xl)) = NaN;
    end

    %find bottom eyelid
    bottomeyelid = uint8(imagepupil((rowp+r):size(imagepupil,1),:));
    lines = findline(bottomeyelid);

    if size(lines,1) > 0

        [xl yl] = linecoords(lines, size(bottomeyelid));
        yl = double(yl)+ irl+rowp+r-2;
        xl = double(xl) + icl-1;

        yla = min(yl);

        y2 = yla:size(eyeimage,1);

        ind4 = sub2ind(size(eyeimage),yl,xl);
        imagewithnoise(uint32(ind4)) = NaN;
        imagewithnoise(round(y2), round(xl)) = NaN;

    end

% %For CASIA, eliminate eyelashes by thresholding
%ref = eyeimage < 80;
%coords = find(ref==1);
%imagewithnoise(coords) = NaN;
imshow(eyeimage)
hold on;
theta = 0 : (2 * pi / 10000) : (2 * pi);
pline_x = circleiris(3)* cos(theta) + circleiris(2);
pline_y = circleiris(3)* sin(theta) + circleiris(1);
%hold off;

%theta = 0 : (2 * pi / 10000) : (2 * pi);
%pline_x = round(circleiris(2))* cos(theta) + round(circleiris(0));
%pline_y = circleiris(2)* sin(theta) + circleiris(1);

plot(pline_x,pline_y,'r-', 'LineWidth', 1);
hold on;
theta = 0 : (2 * pi / 10000) : (2 * pi);
%this part is fr pupil ,and x coordinate is x[1]
pline_x1 = circlepupil(3)* cos(theta) + circlepupil(2);
pline_y1 = circlepupil(3)* sin(theta) + circlepupil(1);
plot(pline_x1,pline_y1,'r-', 'LineWidth', 1);
%pline_x1 = round(circlepupil(2))* cos(theta) + round(circlepupil(0));
%pline_y1 = circlepupil(2)* sin(theta) + circlepupil(1);

%plot(pline_x1,pline_y1,'r-', 'LineWidth', 1);
hold off;
%figure(3)
%imshow(imagewithnoise)
%Folder='C:\Users\HP\Pictures\iris\'
%File='image.jpg'
%imwrite(Figure 1, fullfile(Folder, File));
%disp("heheheh")
%disp(size(eyeimage))

end
function [gradient, or] = canny(im, sigma, scaling, vert, horz)

    xscaling = vert;
    yscaling = horz;

    hsize = [6*sigma+1, 6*sigma+1];   % The filter size.

    gaussian = fspecial('gaussian',hsize,sigma);
    im = filter2(gaussian,im);        % Smoothed image.

    im = imresize(im, scaling);

    [rows, cols] = size(im);

    h =  [  im(:,2:cols)  zeros(rows,1) ] - [  zeros(rows,1)  im(:,1:cols-1)  ];
    v =  [  im(2:rows,:); zeros(1,cols) ] - [  zeros(1,cols); im(1:rows-1,:)  ];
    d1 = [  im(2:rows,2:cols) zeros(rows-1,1); zeros(1,cols) ] - ...
                                   [ zeros(1,cols); zeros(rows-1,1) im(1:rows-1,1:cols-1)  ];
    d2 = [  zeros(1,cols); im(1:rows-1,2:cols) zeros(rows-1,1);  ] - ...
                                   [ zeros(rows-1,1) im(2:rows,1:cols-1); zeros(1,cols)   ];

    X = ( h + (d1 + d2)/2.0 ) * xscaling;
    Y = ( v + (d1 - d2)/2.0 ) * yscaling;

    gradient = sqrt(X.*X + Y.*Y); % Gradient amplitude.

    or = atan2(-Y, X);            % Angles -pi to + pi.
    neg = or<0;                   % Map angles to 0-pi.
    or = or.*~neg + (or+pi).*neg; 
    or = or*180/pi;               % Convert to degrees.
end
function newim = adjgamma(im, g)

        if g <= 0
        error('Gamma value must be > 0');
        end

        if isa(im,'uint8');
        newim = double(im);
        else 
        newim = im;
        end

        % rescale range 0-1
        newim = newim-min(min(newim));
        newim = newim./max(max(newim));

        newim =  newim.^(1/g);   % Apply gamma function
end
function im = nonmaxsup(inimage, orient, radius)

    if size(inimage) ~= size(orient)
      error('image and orientation image are of different sizes');
    end

    if radius < 1
      error('radius must be >= 1');
    end

    [rows,cols] = size(inimage);
    im = zeros(rows,cols);        % Preallocate memory for output image for speed
    iradius = ceil(radius);

    % Precalculate x and y offsets relative to centre pixel for each orientation angle 

    angle = [0:180].*pi/180;    % Array of angles in 1 degree increments (but in radians).
    xoff = radius*cos(angle);   % x and y offset of points at specified radius and angle
    yoff = radius*sin(angle);   % from each reference position.

    hfrac = xoff - floor(xoff); % Fractional offset of xoff relative to integer location
    vfrac = yoff - floor(yoff); % Fractional offset of yoff relative to integer location

    orient = fix(orient)+1;     % Orientations start at 0 degrees but arrays start
                                % with index 1.

    % Now run through the image interpolating grey values on each side
    % of the centre pixel to be used for the non-maximal suppression.

    for row = (iradius+1):(rows - iradius)
      for col = (iradius+1):(cols - iradius) 

        or = orient(row,col);   % Index into precomputed arrays

        x = col + xoff(or);     % x, y location on one side of the point in question
        y = row - yoff(or);

        fx = floor(x);          % Get integer pixel locations that surround location x,y
        cx = ceil(x);
        fy = floor(y);
        cy = ceil(y);
        tl = inimage(fy,fx);    % Value at top left integer pixel location.
        tr = inimage(fy,cx);    % top right
        bl = inimage(cy,fx);    % bottom left
        br = inimage(cy,cx);    % bottom right

        upperavg = tl + hfrac(or) * (tr - tl);  % Now use bilinear interpolation to
        loweravg = bl + hfrac(or) * (br - bl);  % estimate value at x,y
        v1 = upperavg + vfrac(or) * (loweravg - upperavg);

      if inimage(row, col) > v1 % We need to check the value on the other side...

        x = col - xoff(or);     % x, y location on the `other side' of the point in question
        y = row + yoff(or);

        fx = floor(x);
        cx = ceil(x);
        fy = floor(y);
        cy = ceil(y);
        tl = inimage(fy,fx);    % Value at top left integer pixel location.
        tr = inimage(fy,cx);    % top right
        bl = inimage(cy,fx);    % bottom left
        br = inimage(cy,cx);    % bottom right
        upperavg = tl + hfrac(or) * (tr - tl);
        loweravg = bl + hfrac(or) * (br - bl);
        v2 = upperavg + vfrac(or) * (loweravg - upperavg);

        if inimage(row,col) > v2            % This is a local maximum.
          im(row, col) = inimage(row, col); % Record value in the output image.
        end

       end
      end
    end
end
function bw = hysthresh(im, T1, T2)

if (T2 > T1 | T2 < 0 | T1 < 0)  % Check thesholds are sensible
  error('T1 must be >= T2 and both must be >= 0 ');
end

[rows, cols] = size(im);    % Precompute some values for speed and convenience.
rc = rows*cols;
rcmr = rc - rows;
rp1 = rows+1;

bw = im(:);                 % Make image into a column vector
pix = find(bw > T1);        % Find indices of all pixels with value > T1
npix = size(pix,1);         % Find the number of pixels with value > T1

stack = zeros(rows*cols,1); % Create a stack array (that should never
                            % overflow!)

stack(1:npix) = pix;        % Put all the edge points on the stack
stp = npix;                 % set stack pointer
for k = 1:npix
    bw(pix(k)) = -1;        % mark points as edges
end


% Precompute an array, O, of index offset values that correspond to the eight 
% surrounding pixels of any point. Note that the image was transformed into
% a column vector, so if we reshape the image back to a square the indices 
% surrounding a pixel with index, n, will be:
%              n-rows-1   n-1   n+rows-1
%
%               n-rows     n     n+rows
%                     
%              n-rows+1   n+1   n+rows+1

O = [-1, 1, -rows-1, -rows, -rows+1, rows-1, rows, rows+1];

while stp ~= 0            % While the stack is not empty
    v = stack(stp);         % Pop next index off the stack
    stp = stp - 1;
    
    if v > rp1 & v < rcmr   % Prevent us from generating illegal indices
			    % Now look at surrounding pixels to see if they
                            % should be pushed onto the stack to be
                            % processed as well.
       index = O+v;	    % Calculate indices of points around this pixel.	    
       for l = 1:8
	       ind = index(l);
    	   if bw(ind) > T2   % if value > T2,
    	       stp = stp+1;  % push index onto the stack.
    	       stack(stp) = ind;
    	       bw(ind) = -1; % mark this as an edge point
    	   end
       end
    end
end



bw = (bw == -1);            % Finally zero out anything that was not an edge 
bw = reshape(bw,rows,cols); % and reshape the image
end
function [x,y] = linecoords(lines, imsize)

    xd = [1:imsize(2)];
    yd = (-lines(3) - lines(1)*xd ) / lines(2);

    coords = find(yd>imsize(1));
    yd(coords) = imsize(1);
    coords = find(yd<1);
    yd(coords) = 1;

    x = int32(xd);
    y = int32(yd);   
    end
function lines = findline(image)

    [I2 or] = canny(image, 2, 1, 0.00, 1.00);

    I3 = adjgamma(I2, 1.9);
    I4 = nonmaxsup(I3, or, 1.5);
    edgeimage = hysthresh(I4, 0.20, 0.15);


    theta = (0:179)';
    [R, xp] = radon(edgeimage, theta);

    maxv = max(max(R));

    if maxv > 25
        i = find(R == max(max(R)));
    else
        lines = [];
        return;
    end

    [foo, ind] = sort(-R(i));
    u = size(i,1);
    k = i(ind(1:u));
    [y,x]=ind2sub(size(R),k);
    t = -theta(x)*pi/180;
    r = xp(y);

    lines = [cos(t) sin(t) -r];

    cx = size(image,2)/2-1;
    cy = size(image,1)/2-1;
    lines(:,3) = lines(:,3) - lines(:,1)*cx - lines(:,2)*cy;
    end
function sum = ContourIntegralCircular(imagen,y_0,x_0,r,angs)

% EXHAUSTIVE EXTENSIVE ALGORITHM
% rc = r^2;
% sum = 0;
% for x = max(1,x_0-r):min(size(imagen,2),x_0+r)
%     for y = max(1,y_0-r):min(size(imagen,1),y_0+r)
%         if abs((x-x_0)^2+(y-y_0)^2-rc)<2
%             sum = sum + imagen(y,x);
%         end
%     end
% end
% sum = sum/r;

% LIGHT ALGORITHM
    sum = 0;
    for ang = angs
        y = round(y_0-cos(ang)*r);
        x = round(x_0+sin(ang)*r);
        if y<1
            y = 1;
        elseif y>size(imagen,1)
            y = size(imagen,1);
        end
        if x<1
            x = 1;
        elseif x>size(imagen,2)
            x = size(imagen,2);
        end
        sum = sum + imagen(y,x);
    end
end

%function [inner_y inner_x inner_r] = SearchInnerBoundary(imagen)

%fprintf(1,'Searching for inner boundary of the iris \n');

% INTEGRODIFFERENTIAL OPERATOR COARSE (jump-level precision)
%function [rowp, colp, rp]= SearchInnerBoundary(imagen)
function [inner_y inner_x inner_r]= SearchInnerBoundary(imagen)
    Y = size(imagen,1);
    X = size(imagen,2);
    sect = X/4; % SECTor. Width of the external margin for which search is excluded
    minrad = 10;
    maxrad = sect*0.8;
    jump = 4; % precision of the coarse search, in pixels
    hs = zeros(...
        floor((Y-2*sect)/jump),...
        floor((X-2*sect)/jump),...
        floor((maxrad-minrad)/jump)); % Hough Space (y,x,r)
    disp(size(hs))
    integrationprecision = 1; % resolution of the circular integration
    angs = 0:integrationprecision:(2*pi);
    for x = 1:size(hs,2)
        for y = 1:size(hs,1)
            for r = 1:size(hs,3)
                hs(y,x,r) = ContourIntegralCircular(imagen,...
                    sect+y*jump, sect+x*jump, minrad+r*jump, angs);
            end
        end
        %fprintf(1,'column : %d\n',x);
        %fprintf(1,'.');
    end
    disp(hs)
    %fprintf(1,'\n');
    hspdr = hs-hs(: , : , [1 , 1:size(hs,3)-1]); % Hough Space Partial Derivative R

    % BLURRING
    sm = 3; % size of the blurring mask
    hspdrs = convn(hspdr,ones(sm,sm,sm),'same');

    [maxim indmax] = max(hspdrs(:));
    [y,x,r] = ind2sub(size(hspdrs),indmax);
    inner_y = sect + (y)*jump;
    inner_x = sect + (x)*jump;
    inner_r = minrad + (r-1)*jump;


    % INTEGRODIFFERENTIAL OPERATOR FINE (pixel-level precision)
    %jump = jump*2;
    hs = zeros(jump*2,jump*2,jump*2); % Hough Space (y,x,r)
    integrationprecision = 0.1; % resolution of the circular integration
    angs = 0:integrationprecision:(2*pi);
    for x = 1:size(hs,2)
        for y = 1:size(hs,1)
            for r = 1:size(hs,3)
                hs(y,x,r) = ContourIntegralCircular(imagen,...
                    inner_y-jump+y, inner_x-jump+x, inner_r-jump+r, angs);
            end
        end
        %fprintf(1,'column : %d\n',x);
    %     fprintf(1,'.');
    end
    % fprintf(1,'\n');
    hspdr = hs - hs(: , : , [1 , 1 : size(hs,3)- 1 ]); % Hough Space Partial Derivative R

    % BLURRING
    sm = 3; % size of the blurring mask
    hspdrs = convn(hspdr,ones(sm,sm,sm),'same');

    [maxim indmax] = max(hspdrs(:));
    [y,x,r] = ind2sub(size(hspdrs),indmax);
    inner_y = inner_y-jump+y;
    inner_x = inner_x-jump+x;
    inner_r = inner_r-jump+r-1;
end

function [outer_y outer_x outer_r] = SearchOuterBoundary(imagen, inner_y, inner_x, inner_r)

%fprintf(1,'Searching for outer boundary of the iris \n');

% INTEGRODIFFERENTIAL OPERATOR
    maxdispl = round(inner_r*0.15); % very maximum displacement 15% (Daugman 2004)
    minrad = round(inner_r/0.8); maxrad = round(inner_r/0.3); %0.1-0.8 (Daugman 2004)
    hs = zeros(2*maxdispl,2*maxdispl,maxrad-minrad); % Hough Space (y,x,r)
    intreg = [2/6 4/6; 8/6 10/6]*pi; % integration region, avoiding eyelids
    %intreg = [1/4 3/4; 5/4 7/4]*pi;
    integrationprecision = 0.05; % resolution of the circular integration
    angs = [intreg(1,1):integrationprecision:intreg(1,2) intreg(2,1):integrationprecision:intreg(2,2)];
    for x = 1:size(hs,2)
        for y = 1:size(hs,1)
            for r = 1:size(hs,3)
                hs(y,x,r) = ContourIntegralCircular(imagen,...
                    inner_y-maxdispl+y, inner_x-maxdispl+x, minrad+r, angs);
            end
        end
        %fprintf(1,'column : %d\n',x)
        %fprintf(1,'.');
    end
    %fprintf(1,'\n');
    hspdr = hs-hs(:,:,[1,1:size(hs,3)-1]); % Hough Space Partial Derivative R

    % BLURRING
    sm = 7; % size of the blurring mask
    hspdrs = convn(hspdr,ones(sm,sm,sm),'same');

    [maxim indmax] = max(hspdrs(:));
    [y,x,r] = ind2sub(size(hspdrs),indmax);
    outer_y = inner_y - maxdispl + y;
    outer_x = inner_x - maxdispl + x;
    outer_r = minrad + r - 1;
end
%link:
%https://github.com/AntiAegis/Iris-Recognition/tree/master/matlab/fnc




















